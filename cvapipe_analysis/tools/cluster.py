import dask
from pathlib import Path
from datetime import datetime
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

def get_distributed_executor_address_on_slurm(log, config):

    # Forces a distributed cluster instantiation
    log_dir_name = datetime.now().isoformat().split(".")[0]
    log_dir = Path(f".dask_logs/{log_dir_name}").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure dask config
    dask.config.set(
        {
            "scheduler.work-stealing": False,
            "logging.distributed.worker": "info",
        }
    )

    # Create cluster
    log.info("Creating SLURMCluster")
    slurm_cluster = SLURMCluster(
        cores=config["resources"]["cores"],
        memory=config["resources"]["memory"],
        queue=config["resources"]["queue"],
        walltime=config["resources"]["walltime"],
        local_directory=str(log_dir),
        log_directory=str(log_dir),
    )

    # Spawn workers
    slurm_cluster.scale(jobs=config["resources"]["nworkers"])
    log.info("Created SLURMCluster")
    client = Client(slurm_cluster)
    print(client.get_versions(check=True))

    # Use the port from the created connector to set executor address
    distributed_executor_address = slurm_cluster.scheduler_address

    log.info(f"Dask dashboard available at: {slurm_cluster.dashboard_link}")

    return distributed_executor_address
