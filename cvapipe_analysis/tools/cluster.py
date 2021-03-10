import json
import dask
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

from cvapipe_analysis.tools import shapespace
from cvapipe_analysis.steps.shapemode.avgshape import digitize_shape_mode

class Distribute:
    
    jobs = []
    rel_path_to_dataframe = None
    rel_path_to_input_images = None
    rel_path_to_output = None
    rel_path_to_python_file = None
    folders = ['log','scripts','config','dataframes']
    
    def __init__(self, df: pd.DataFrame, nworkers: int):
        self.df = df.copy().reset_index()
        self.nrows = len(df)
        self.nworkers = nworkers
        if nworkers is not None:
            self.chunk_size = self.nrows // self.nworkers
        self.root = Path().absolute()
        self.root_as_str = str(self.root)
        self.abs_path_to_script_as_str = f"{self.root_as_str}/.distribute/scripts/jobs.sh"
        self.abs_path_jobs_file_as_str = f"{self.root_as_str}/.distribute/scripts/jobs.txt"    
        
    def get_next_chunk(self):
        
        if self.nworkers is None:
            raise ValueError("Number of workers have to be defined for chunk iteration.")
        
        for chunk, df_chunk in self.df.groupby(np.arange(self.nrows)//self.chunk_size):
            id_ini = int(df_chunk.index[0])
            id_end = int(df_chunk.index[-1])
            nrows = id_end - id_ini + 1
            yield chunk, id_ini, nrows

            
    def set_rel_path_to_dataframe(self, path):
        self.rel_path_to_dataframe = path

        
    def get_abs_path_to_dataframe_as_str(self):
        path = self.root / self.rel_path_to_dataframe
        return str(path)

    
    def set_rel_path_to_input_images(self, path):
        self.rel_path_to_input_images = path

        
    def get_abs_path_to_input_images_as_str(self):
        path = self.root / self.rel_path_to_input_images
        return str(path)
        
        
    def set_rel_path_to_output(self, path):
        self.rel_path_to_output = path

        
    def get_abs_path_to_output_as_str(self):
        path = self.root / self.rel_path_to_output
        return str(path)

    
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
        abs_path_python_env = config['resources']['path_python_env']
        with open(self.abs_path_jobs_file_as_str, "w") as fs:
            for job in self.jobs:
                abs_path_config_file = f"{self.root_as_str}/.distribute/config/{job}.json"
                print(f"{abs_path_python_env} {self.get_abs_path_to_python_file_as_str()} --config {abs_path_config_file}", file=fs)
    
    def write_script_file(self, config):
        mem = config['resources']['memory']
        cores = config['resources']['cores']
        nworkers = config['resources']['nworkers']
        abs_path_output_folder = f"{self.root_as_str}/.distribute/log"
        
        with open(self.abs_path_to_script_as_str, "w") as fs:
            print("#!/bin/bash", file=fs)
            print("#SBATCH --partition aics_cpu_general", file=fs)
            print(f"#SBATCH --mem-per-cpu {mem}", file=fs)
            print(f"#SBATCH --cpus-per-task {cores}", file=fs)
            print(f"#SBATCH --output {abs_path_output_folder}/%A_%a.out", file=fs)
            print(f"#SBATCH --error {abs_path_output_folder}/%A_%a.err", file=fs)
            print(f"#SBATCH --array=1-{len(self.jobs)}%{nworkers}", file=fs)
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

        for chunk, id_ini, nrows in self.get_next_chunk():

            script_config = {
                "csv": self.get_abs_path_to_dataframe_as_str(),
                "output": self.get_abs_path_to_output_as_str(),
                "data_folder": self.get_abs_path_to_input_images_as_str(),
                "skip": id_ini,
                "nrows": nrows
            }

            rel_path_config_file = f".distribute/config/{chunk}.json"
            with open(rel_path_config_file, "w") as fj:
                json.dump(script_config, fj, indent=4, sort_keys=True)

            self.append_job(chunk)
            
        self.execute(config, log)

        return

class DistributeFeatures(Distribute):
    def __init__(self, df: pd.DataFrame, nworkers: int):
        super().__init__(df, nworkers)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/compute_features/compute_features_tools.py"

        
class DistributeParameterization(Distribute):
    def __init__(self, df: pd.DataFrame, nworkers: int):
        super().__init__(df, nworkers)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/parameterization/parameterization_tools.py"
        
        
class DistributeAggregation(Distribute):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df, None)
        self.rel_path_to_python_file = "cvapipe_analysis/steps/aggregation/aggregation_tools.py"

        
    def set_rel_path_to_shapemode_results(self, path):
        self.rel_path_to_shapemode_results = path

        
    def get_abs_path_to_shapemode_results_as_str(self):
        path = self.root / self.rel_path_to_shapemode_results
        return str(path)

    
    def check_output_file_exist(self, job):
        path = Path(self.get_abs_path_to_output_as_str()) / f"{job}.tif"
        return path.is_file()

    
    def distribute(self, config, log):
        
        log.info("Cleaning distribute directory.")
        self.clean_distribute_folder()

        PREFIX = config['aggregation']['aggregate_on']
        
        pc_names = [f for f in self.df.columns if PREFIX in f]

        space = shapespace.ShapeSpace(self.df[pc_names])
        
        for pc_name in tqdm(pc_names):
            
            space.set_active_axis(pc_name)
            space.digitize_active_axis()
            space.link_results_folder(Path(self.get_abs_path_to_shapemode_results_as_str()))
            
            for _, intensity in tqdm(config['parameterization']['intensities'], leave=False):                
                for agg in tqdm(config['aggregation']['type'], leave=False):
                    for b, _ in space.iter_map_points():
                        
                        indexes = space.get_indexes_in_bin(b)
                        
                        for struct in tqdm(config['structures']['genes'], leave=False):
                            
                            df_struct = self.df.loc[(self.df.index.isin(indexes))&(self.df.structure_name==struct)]
                            
                            script_id = f"{agg}-{intensity}-{struct}-{pc_name}-B{b}"
                            
                            if (len(df_struct)>0) & (not self.check_output_file_exist(script_id)):
                                
                                path_to_manifest = f".distribute/dataframes/{script_id}.csv"
                                self.set_rel_path_to_dataframe(path_to_manifest)
                                df_struct.to_csv(path_to_manifest)

                                script_config = {
                                    "csv": self.get_abs_path_to_dataframe_as_str(),
                                    "shapemode_results": self.get_abs_path_to_shapemode_results_as_str(),
                                    "output": self.get_abs_path_to_output_as_str(),
                                    "filename": f"{script_id}.tif"
                                }

                                rel_path_config_file = f".distribute/config/{script_id}.json"
                                with open(rel_path_config_file, "w") as fj:
                                    json.dump(script_config, fj, indent=4, sort_keys=True)

                                self.append_job(script_id)

        self.execute(config, log)
        
        return
