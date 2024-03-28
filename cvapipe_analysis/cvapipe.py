import os
import argparse
from cvapipe_analysis.tools import printer

print = printer.Printer().cprint

def main():

    print("Starting cvapipe analysis workflow...")

    parser = argparse.ArgumentParser(description="CVAPIPE analysis workflow.")
    parser.add_argument("--step", help="Step name.", required=True)
    parser.add_argument("--staging", help="Path to staging folder.", required=True)
    parser.add_argument("--mode", help="Mode of data loading. See help for details.", required=True)
    args = vars(parser.parse_args())

    step = args["step"]
    staging = args["staging"]

    if step == "load_data":
        mode = args["mode"]
        cmd = f"cvapipe_load_data --mode {mode} --staging {staging}"
        os.system(cmd)

    # path = general.get_path_to_default_config()
    # config = general.load_config_file(path)
    # config["project"]["local_staging"] = staging
    # control = controller.Controller(config)

    # loader = DataLoader(control)
    # if ignore_raw_data:
    #     loader.disable_download_of_raw_data()
    # df = loader.load(kwargs)

    # path = Path(staging)
    # df.to_csv(path/f"{self.step_name}/manifest.csv")
    # general.save_config(config, path)
    # return
