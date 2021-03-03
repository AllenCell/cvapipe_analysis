import yaml
import json
    
def load_config_file():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    return config

def create_workflow_file_from_config():
    config = load_config_file()
    suffix = config['project']['name']
    with open("workflow_config.json", "w") as fj:
        json.dump({"project_local_staging_dir": f"local_staging_{suffix}"}, fj)
