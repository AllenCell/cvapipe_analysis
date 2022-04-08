import json
import yaml
import sys
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from cvapipe_analysis.tools import controller


def load_config_file(path: Path="./"):
    with open(Path(path)/'config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def write_workflow_config(config):
    local_staging = config['local_staging']
    with open('workflow_config.json', 'w') as fj:
        json.dump({'project_local_staging_dir': local_staging}, fj)

def resolve_config():
    local_staging = sys.argv[3]
    
    config_path = Path(local_staging) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Configuration file config.yaml not found")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["local_staging"] = local_staging
    config["project_local_staging_dir"] = local_staging
    write_workflow_config(config)
    return config, config_path


def save_config_file(path_to_folder, config):
    path_to_folder = Path(path_to_folder)
    shutil.copyfile(config, path_to_folder/"parameters.yaml")
    return


def get_date_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@contextmanager
def configuration(config, config_path=None, staging_dir=None):
    control = controller.Controller(config)
    try:
        control.log({"start": get_date_time()})
        yield control
    finally:
        if staging_dir is not None and config_path is not None:
            control.log({"end": get_date_time()})
            save_config_file(staging_dir, config_path)

