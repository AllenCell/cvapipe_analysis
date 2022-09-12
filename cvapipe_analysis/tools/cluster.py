import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cvapipe_analysis


class Distributor:
    """
    The goal of this class is to provide an interface
    to spawn many jobs in slurm using array of jobs.
    A master dataframe is provided in which each row
    corresponds to one job that will run. Each row
    contains all information necessary for that job to
    run. The class provides prvides a way of iterating
    over chunks of the dataframe. The number of chunks
    depends on the number of workers requested.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, step, control):
        self.jobs = []
        self.step_name = step.step_name
        self.control = control
        self.stepfolders = ['log', 'dataframes']
        self.nworkers = control.get_distributed_number_of_workers()
        self.abs_path_to_distribute = control.get_staging() / f".distribute-{self.step_name}"
        self.abs_path_to_script_as_str = str(self.abs_path_to_distribute / "jobs.sh")
        self.abs_path_jobs_file_as_str = str(self.abs_path_to_distribute / "jobs.txt")
        self.abs_path_to_cvapipe = Path(
            os.path.abspath(cvapipe_analysis.__file__)).parents[1]

    def set_data(self, df):
        self.df = df
        self.nrows = len(df)
        self.chunk_size = round(0.5 + self.nrows / self.nworkers)

    def set_chunk_size(self, n):
        self.chunk_size = n

    def get_next_chunk(self):
        for chunk, df_chunk in self.df.groupby(np.arange(self.nrows) // self.chunk_size):
            yield chunk, df_chunk
    
    def get_next_block(self):
        block = 0
        ni = int(np.sqrt(self.nworkers))
        nj = int(self.nworkers / ni)
        for indexes_i in np.array_split(self.df.index.values, ni):
            for indexes_j in np.array_split(self.df.index.values, nj):
                col1 = np.hstack([indexes_i, indexes_j])
                col2 = np.hstack([0*indexes_i, 1+0*indexes_j])
                df = pd.DataFrame(np.vstack([col1, col2]).T, columns=["CellId", "Group"])
                df = df.set_index("CellId")
                block += 1
                yield block, df

    def get_abs_path_to_python_file_as_str(self):
        path = self.abs_path_to_cvapipe / self.rel_path_to_python_file
        return str(path)

    def clean_distribute_folder(self):
        for folder in self.stepfolders:
            path_subfolder = self.abs_path_to_distribute / folder
            try:
                shutil.rmtree(str(path_subfolder))
            except:
                pass
            path_subfolder.mkdir(parents=True, exist_ok=True)
        return

    def append_job(self, job):
        self.jobs.append(job)

    def write_commands_file(self):
        crtl = self.control
        python__env_path_as_str = crtl.get_distributed_python_env_as_str()
        with open(self.abs_path_jobs_file_as_str, "w") as fs:
            for job in self.jobs:
                abs_path_to_dataframe = str(
                    self.abs_path_to_distribute / f"dataframes/{job}.csv")
                print(
                    f"{python__env_path_as_str} {self.get_abs_path_to_python_file_as_str()} --csv {abs_path_to_dataframe}", file=fs)

    def write_script_file(self):
        crtl = self.control
        abs_path_output_folder = self.abs_path_to_distribute / "log"
        with open(self.abs_path_to_script_as_str, "w") as fs:
            print("#!/bin/bash", file=fs)
            print("#SBATCH --partition aics_cpu_general", file=fs)
            print(f"#SBATCH --mem-per-cpu {crtl.get_distributed_memory()}", file=fs)
            print(f"#SBATCH --cpus-per-task {crtl.get_distributed_cores()}", file=fs)
            print(f"#SBATCH --output {abs_path_output_folder}/%A_%a.out", file=fs)
            print(f"#SBATCH --error {abs_path_output_folder}/%A_%a.err", file=fs)
            print(f"#SBATCH --array=1-{len(self.jobs)}", file=fs)
            print("#SBATCH --job-name=cvapipe", file=fs)
            print(
                f"srun $(head -n $SLURM_ARRAY_TASK_ID {self.abs_path_jobs_file_as_str} | tail -n 1)", file=fs)
        return

    def execute(self):
        self.write_commands_file()
        self.write_script_file()
        print(f"Submitting {len(self.jobs)} cvapipe_analysis jobs...")
        submission = 'sbatch ' + self.abs_path_to_script_as_str
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
        process.communicate()
        return

    def distribute(self):
        print("\nCleaning distribute directory.\n")
        self.clean_distribute_folder()
        print(f"\nDistributing dataframe of shape: {self.df.shape}")
        print(f"in chunks of size {self.chunk_size}.\n")
        for chunk, df_chunk in self.get_next_chunk():
            abs_path_to_dataframe = self.abs_path_to_distribute/f"dataframes/{chunk}.csv"
            df_chunk.to_csv(abs_path_to_dataframe)
            self.append_job(chunk)
        self.execute()
        return

    def distribute_by_row(self):
        print("\nCleaning distribute directory.\n")
        self.clean_distribute_folder()
        print(f"\nDistributing dataframe of shape: {self.df.shape} by row.")
        for chunk, (_, row) in enumerate(self.df.iterrows()):
            abs_path_to_dataframe = self.abs_path_to_distribute/f"dataframes/{chunk}.csv"
            pd.DataFrame(row).to_csv(abs_path_to_dataframe)
            self.append_job(chunk)
        self.execute()
        return

    def distribute_by_blocks(self):
        print("\nCleaning distribute directory.\n")
        self.clean_distribute_folder()
        print(f"\nDistributing dataframe of shape: {self.df.shape}")
        print(f"in {self.nworkers} blocks.\n")
        for block, df_block in self.get_next_block():
            abs_path_to_dataframe = self.abs_path_to_distribute/f"dataframes/{block}.csv"
            df_block.to_csv(abs_path_to_dataframe)
            self.append_job(block)
        self.execute()
        return

class FeaturesDistributor(Distributor):
    def __init__(self, step, control):
        super().__init__(step, control)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/compute_features/compute_features_tools.py"


class ParameterizationDistributor(Distributor):
    def __init__(self, step, control):
        super().__init__(step, control)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/parameterization/parameterization_tools.py"


class AggregationDistributor(Distributor):
    def __init__(self, step, control):
        super().__init__(step, control)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/aggregation/aggregation_tools.py"


class StereotypyDistributor(Distributor):
    def __init__(self, step, control):
        super().__init__(step, control)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/stereotypy/stereotypy_tools.py"


class ConcordanceDistributor(Distributor):
    def __init__(self, step, control):
        super().__init__(step, control) 
        self.rel_path_to_python_file = "cvapipe_analysis/steps/concordance/concordance_tools.py"


class CorrelationDistributor(Distributor):
    # Requires blocks distribution instead of chunks
    def __init__(self, step, control):
        super().__init__(step, control)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/correlation/correlation_tools.py"
