import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from cvapipe_analysis.tools import controller

def load_config_file(path: Path="./", fname="config.yaml"):
    with open(Path(path)/fname, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_config_file(path_to_folder, filename="parameters.yaml"):
    print("WARNING: This function is deprecated. Please use save_config instead.")
    return
    path_to_folder = Path(path_to_folder)
    shutil.copyfile("./config.yaml", path_to_folder/filename)
    return

def save_config(config, path, filename="config.yaml"):
    with open(path/filename, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def get_date_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def create_workflow_file_from_config(staging):
    with open("workflow_config.json", "w") as fj:
        json.dump({"project_local_staging_dir": staging}, fj)

@contextmanager
def configuration(path=None):
    config = load_config_file(path=path)
    control = controller.Controller(config)
    try:
        control.log({"start": get_date_time()})
        yield control
    finally:
        if path is not None:
            control.log({"end": get_date_time()})
            save_config(config, Path(path), filename="parameters.yaml")
