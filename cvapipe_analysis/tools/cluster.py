import json
import dask
import shutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def run_distributed_feature_extraction(df, path_df, data_folder, output_folder, config, log):

    log.info("Cleaning distribute directory.")
    
    try:
        shutil.rmtree('.distribute')
    except: pass
    
    for folder in ['log','scripts','config']:
        dir_dist = Path(".distribute") / folder
        dir_dist.mkdir(parents=True, exist_ok=True)

    nrows = len(df)
    nworkers = config['resources']['nworkers']
    
    df = df.reset_index(drop=False)
    
    log.info("Creating config files.")
    
    root = Path().absolute()
    
    chunk_size = len(df) // nworkers
    
    for chunk, df_chunk in df.groupby(np.arange(nrows)//chunk_size):

        id_ini = int(df_chunk.index[0])
        id_end = int(df_chunk.index[-1])
        nrows = id_end - id_ini + 1
        
        # Script config
        
        script_config = {
            "csv": str(root / path_df),
            "skip": id_ini,
            "nrows": nrows,
            "data_folder": str(root / data_folder),
            "output": str(root / output_folder)
        }
            
        with open(f".distribute/config/{chunk}.json", "w") as fj:
            json.dump(script_config, fj, indent=4, sort_keys=True)
    
        # Script

        script_name = f".distribute/scripts/{chunk}.script"
        
        with open(script_name, "w") as fs:
            print("#!/bin/bash", file=fs)
            print(f"#SBATCH --job-name=cvapipe-{chunk}", file=fs)
            print("#SBATCH --partition aics_cpu_general", file=fs)
            print(f"#SBATCH --mem-per-cpu {config['resources']['memory']}", file=fs)
            print(f"#SBATCH --cpus-per-task {config['resources']['cores']}", file=fs)
            print(f"#SBATCH --output {str(root)}/.distribute/log/{chunk}.out", file=fs)
            print(f"srun {config['resources']['path_python_env']} {str(root)}/cvapipe_analysis/steps/compute_features/compute_features_tools.py --config {str(root)}/.distribute/config/{chunk}.json", file=fs)
    
        log.info(f"Submitting job cvapipe {chunk}...")

        submission = 'sbatch ' + script_name
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
        (out, err) = process.communicate()

    return