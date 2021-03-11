import os
import json
import dask
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

import cvapipe_analysis
from cvapipe_analysis.tools import shapespace
from cvapipe_analysis.steps.shapemode.avgshape import digitize_shape_mode

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
    jobs = []
    folders = ['log','dataframes']
    
    def __init__(self, df, nworkers):
        self.df = df.copy().reset_index()
        self.nrows = len(df)
        self.nworkers = nworkers
        self.chunk_size = round(0.5+self.nrows/self.nworkers)
        self.root = Path(os.path.abspath(cvapipe_analysis.__file__)).parents[1]
        self.root_as_str = str(self.root)
        self.abs_path_to_script_as_str = f"{self.root_as_str}/.distribute/jobs.sh"
        self.abs_path_jobs_file_as_str = f"{self.root_as_str}/.distribute/jobs.txt"    
        
    def get_next_chunk(self):
        for chunk, df_chunk in self.df.groupby(np.arange(self.nrows)//self.chunk_size):
            yield chunk, df_chunk

    def get_abs_path_to_python_file_as_str(self):
        path = self.root / self.rel_path_to_python_file
        return str(path)

    def clean_distribute_folder(self):
        for folder in self.folders:
            path_subfolder = Path(".distribute") / folder
            try:
                shutil.rmtree(str(path_subfolder))
            except: pass
            path_subfolder.mkdir(parents=True, exist_ok=True)
        return

    def append_job(self, job):
        self.jobs.append(job)
    
    def write_commands_file(self, config):
        python_path_as_str = "/"+config['resources']['python_env'].lstrip('/')+"/bin/python"
        with open(self.abs_path_jobs_file_as_str, "w") as fs:
            for job in self.jobs:
                abs_path_to_dataframe = f"{self.root_as_str}/.distribute/dataframes/{job}.csv"
                print(f"{python_path_as_str} {self.get_abs_path_to_python_file_as_str()} --csv {abs_path_to_dataframe}", file=fs)
    
    def write_script_file(self, config):
        abs_path_output_folder = f"{self.root_as_str}/.distribute/log"
        with open(self.abs_path_to_script_as_str, "w") as fs:
            print("#!/bin/bash", file=fs)
            print("#SBATCH --partition aics_cpu_general", file=fs)
            print(f"#SBATCH --mem-per-cpu {config['resources']['memory']}", file=fs)
            print(f"#SBATCH --cpus-per-task {config['resources']['cores']}", file=fs)
            print(f"#SBATCH --output {abs_path_output_folder}/%A_%a.out", file=fs)
            print(f"#SBATCH --error {abs_path_output_folder}/%A_%a.err", file=fs)
            print(f"#SBATCH --array=1-{len(self.jobs)}", file=fs)
            print(f"srun $(head -n $SLURM_ARRAY_TASK_ID {self.abs_path_jobs_file_as_str} | tail -n 1)", file=fs)

        return

    def execute(self, config, log):
        self.write_commands_file(config)
        self.write_script_file(config)
        
        log.info(f"Submitting {len(self.jobs)} cvapipe_analysis jobs...")
        submission = 'sbatch ' + self.abs_path_to_script_as_str
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
        (out, err) = process.communicate()

    def distribute(self, config, log):
        log.info("Cleaning distribute directory.")
        self.clean_distribute_folder()

        for chunk, df_chunk in self.get_next_chunk():
            rel_path_to_dataframe = f".distribute/dataframes/{chunk}.csv"
            df_chunk.to_csv(rel_path_to_dataframe)
            self.append_job(chunk)
        self.execute(config, log)

        return

class FeaturesDistributor(Distributor):
    def __init__(self, df, nworkers):
        super().__init__(df, nworkers)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/compute_features/compute_features_tools.py"

class ParameterizationDistributor(Distributor):
    def __init__(self, df, nworkers):
        super().__init__(df, nworkers)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/parameterization/parameterization_tools.py"
    
class AggregationDistributor(Distributor):
    def __init__(self, df, nworkers):
        super().__init__(df, nworkers)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/aggregation/aggregation_tools.py"
