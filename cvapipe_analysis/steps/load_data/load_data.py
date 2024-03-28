#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from cvapipe_analysis.tools import general, controller, printer
from cvapipe_analysis.steps.load_data.load_data_tools import DataLoader

print = printer.Printer().cprint

def main():

    print("Running load data step.", 1)

    parser = argparse.ArgumentParser(description="CVAPIPE analysis workflow.")
    parser.add_argument("--staging", help="Path to staging folder.", required=True)
    parser.add_argument("--mode", help="Mode of data loading. See help for details.", required=True)
    args = vars(parser.parse_args())

    print(f"Staging folder: {args['staging']}", 2)
    print(f"Load data mode: {args['mode']}", 2)

    path = general.get_path_to_default_config()
    config = general.load_config_file(path, new_staging=args["staging"])
    control = controller.Controller(config)

    loader = DataLoader(control)
    # if ignore_raw_data:
    #     loader.disable_download_of_raw_data()
    # df = loader.load(args["mode"]) << NEXT IS HOW TO IMPLEMENT CORRECT LOAD ACCORDING TO MODE

    # path = Path(staging)
    # df.to_csv(path/f"{self.step_name}/manifest.csv")
    # general.save_config(config, path)
    # return
