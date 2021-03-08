import json
import dask
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

from cvapipe_analysis.steps.shapemode.avgshape import digitize_shape_mode

class Distribute:
    
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
        
    def write_script_file(self, script_id, config):

        mem = config['resources']['memory']
        cores = config['resources']['cores']
    
        abs_path_to_script = f"{self.root_as_str}/.distribute/scripts/{script_id}.script"
        abs_path_to_script_output = f"{self.root_as_str}/.distribute/log/{script_id}.out"
        abs_path_python_env = config['resources']['path_python_env']
        abs_path_config_file = f"{self.root_as_str}/.distribute/config/{script_id}.json"

        with open(abs_path_to_script, "w") as fs:
            print("#!/bin/bash", file=fs)
            print(f"#SBATCH --job-name=cvapipe-{script_id}", file=fs)
            print("#SBATCH --partition aics_cpu_general", file=fs)
            print(f"#SBATCH --mem-per-cpu {mem}", file=fs)
            print(f"#SBATCH --cpus-per-task {cores}", file=fs)
            print(f"#SBATCH --output {abs_path_to_script_output}", file=fs)
            print(f"srun {abs_path_python_env} {self.get_abs_path_to_python_file_as_str()} --config {abs_path_config_file}", file=fs)

        return abs_path_to_script

    
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

            abs_path_script = self.write_script_file(chunk, config)

            log.info(f"Submitting job cvapipe {chunk}...")

            submission = 'sbatch ' + abs_path_script
            process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
            (out, err) = process.communicate()

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

        
    def set_rel_path_to_shapemode_dataframe(self, path):
        self.rel_path_to_shapemode_dataframe = path

        
    def get_abs_path_to_shapemode_dataframe_as_str(self):
        path = self.root / self.rel_path_to_shapemode_dataframe
        return str(path)


    def distribute(self, config, log):
        
        log.info("Cleaning distribute directory.")
        self.clean_distribute_folder()

        PREFIX = config['aggregation']['aggregate_on']
        
        pc_names = [f for f in self.df.columns if PREFIX in f]

        pc_names = pc_names[:1]
        
        df_shapemode = pd.read_csv(self.get_abs_path_to_shapemode_dataframe_as_str(), index_col=0)
        log.info(f"Shape of shape mode paths manifest: {df_shapemode.shape}")

        for pc_name in tqdm(pc_names):
            
            for _, intensity in tqdm(config['parameterization']['intensities'], leave=False):
                
                for agg in tqdm([('avg', np.mean), ('std', np.std)], leave=False):
                    
                    # Find the indexes of cells in each bin
                    df_filtered, bin_indexes, _ = digitize_shape_mode(
                        df=self.df,
                        feature=pc_name,
                        nbins=config['pca']['number_map_points'],
                        filter_based_on=pc_names
                    )
                    
                    for b in tqdm(range(1, 1+config['pca']['number_map_points']), leave=False):
                        
                        for struct in tqdm(config['structures']['genes'], leave=False):
                            
                            df_struct = df_filtered.loc[df_filtered.structure_name==struct]
                            
                            script_id = f"{agg[0]}-{intensity}-{struct}-{pc_name}-B{b}"
                            
                            df_struct_bin = df_struct.loc[df_struct.index.isin(bin_indexes[b-1][1])]

                            if len(df_struct_bin) > 0:

                                path_to_manifest = f".distribute/dataframes/{script_id}.csv"
                                self.set_rel_path_to_dataframe(path_to_manifest)
                                df_struct_bin.to_csv(path_to_manifest)

                                script_config = {
                                    "csv": self.get_abs_path_to_dataframe_as_str(),
                                    "csv_shapemode": self.get_abs_path_to_shapemode_dataframe_as_str(),
                                    "output": self.get_abs_path_to_output_as_str(),
                                    "filename": f"{script_id}.tif"
                                }

                                rel_path_config_file = f".distribute/config/{script_id}.json"
                                with open(rel_path_config_file, "w") as fj:
                                    json.dump(script_config, fj, indent=4, sort_keys=True)

                                abs_path_script = self.write_script_file(script_id, config)

                                log.info(f"Submitting job cvapipe {script_id}...")

                                submission = 'sbatch ' + abs_path_script
                                process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
                                (out, err) = process.communicate()
        
        return
