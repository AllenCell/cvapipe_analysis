import json
import yaml
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from cvapipe_analysis.tools import controller


def load_config_file(path: Path=Path("./")):
    with open(path/'config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config_file(config, path_to_folder):
    path_to_folder = Path(path_to_folder)
    with open(path_to_folder / 'parameters.yaml', 'w') as f:
        yaml.dump(config, f)
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
            save_config_file(control.config, path_to_folder)
