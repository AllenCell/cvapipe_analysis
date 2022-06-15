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


def save_config_file(path_to_folder):
    path_to_folder = Path(path_to_folder)
    shutil.copyfile("./config.yaml", path_to_folder/"parameters.yaml")
    return


def create_workflow_file_from_config():
    config = load_config_file()
    local_staging = config['project']['local_staging']
    with open('workflow_config.json', 'w') as fj:
        json.dump({'project_local_staging_dir': local_staging}, fj)


def get_date_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@contextmanager
def configuration(path_to_folder=None):
    config = load_config_file()
    control = controller.Controller(config)
    try:
        control.log({"start": get_date_time()})
        yield control
    finally:
        if path_to_folder is not None:
            control.log({"end": get_date_time()})
            save_config_file(path_to_folder)
