import json
import dask
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict


class data_to_distribute:
    
    
    def __init__(self, df: pd.DataFrame, nworkers: int):
        self.df = df.copy().reset_index()
        self.nrows = len(df)
        self.nworkers = nworkers
        self.chunk_size = self.nrows // self.nworkers

        
    def get_next_chunk(self):
        for chunk, df_chunk in self.df.groupby(np.arange(self.nrows)//self.chunk_size):
            id_ini = int(df_chunk.index[0])
            id_end = int(df_chunk.index[-1])
            nrows = id_end - id_ini + 1
            yield chunk, id_ini, nrows

            
    def set_rel_path_to_dataframe(self, path):
        self.rel_path_dataframe = path

        
    def get_abs_path_to_dataframe_as_str(self):
        path = Path().absolute() / self.rel_path_dataframe
        return str(path)

    
    def set_rel_path_to_input_images(self, path):
        self.rel_path_to_input_images = path

        
    def get_abs_path_to_input_images_as_str(self):
        path = Path().absolute() / self.rel_path_to_input_images
        return str(path)
        
        
    def set_rel_path_to_output(self, path):
        self.rel_path_to_output = path

        
    def get_abs_path_to_output_as_str(self):
        path = Path().absolute() / self.rel_path_to_output
        return str(path)

    
def clean_distribute_folder():
    
    folders = ['log','scripts','config']
        
    for folder in folders:
        path_subfolder = Path(".distribute") / folder
        try:
            shutil.rmtree(str(path_subfolder))
        except: pass
        path_subfolder.mkdir(parents=True, exist_ok=True)

    return

def write_script_file(chunk, config, rel_path_python_file):

    mem = config['resources']['memory']
    cores = config['resources']['cores']
    
    root = str(Path().absolute())
    abs_path_script = f"{root}/.distribute/scripts/{chunk}.script"
    abs_path_script_output = f"{root}/.distribute/log/{chunk}.out"
    abs_path_python_env = config['resources']['path_python_env']
    abs_path_python_file = f"{root}/{rel_path_python_file}"
    abs_path_config_file = f"{root}/.distribute/config/{chunk}.json"

    with open(abs_path_script, "w") as fs:
        print("#!/bin/bash", file=fs)
        print(f"#SBATCH --job-name=cvapipe-{chunk}", file=fs)
        print("#SBATCH --partition aics_cpu_general", file=fs)
        print(f"#SBATCH --mem-per-cpu {mem}", file=fs)
        print(f"#SBATCH --cpus-per-task {cores}", file=fs)
        print(f"#SBATCH --output {abs_path_script_output}", file=fs)
        print(f"srun {abs_path_python_env} {abs_path_python_file} --config {abs_path_config_file}", file=fs)

    return abs_path_script


def distribute_python_code(data, config, log, rel_path_python_file):
    
    log.info("Cleaning distribute directory.")
    clean_distribute_folder()
    
    for chunk, id_ini, nrows in data.get_next_chunk():
        
        script_config = {
            "csv": data.get_abs_path_to_dataframe_as_str(),
            "output": data.get_abs_path_to_output_as_str(),
            "data_folder": data.get_abs_path_to_input_images_as_str(),
            "skip": id_ini,
            "nrows": nrows
        }

        rel_path_config_file = f".distribute/config/{chunk}.json"
        with open(rel_path_config_file, "w") as fj:
            json.dump(script_config, fj, indent=4, sort_keys=True)

        abs_path_script = write_script_file(chunk, config, rel_path_python_file)
    
        log.info(f"Submitting job cvapipe {chunk}...")

        submission = 'sbatch ' + abs_path_script
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)
        (out, err) = process.communicate()
        
    return
