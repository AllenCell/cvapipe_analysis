#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will convert all the steps into CLI callables.

You should not edit this script.
"""

import sys
import inspect
import logging

import fire

from cvapipe_analysis import steps
from cvapipe_analysis import tools
from cvapipe_analysis.bin.all import All


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

# Creates a json file for datastep to know where the staging
# folder is located. This is used for datastep to store its
# own information -- not really important for cvapipe.
args = sys.argv
if "--staging" not in args:
    raise ValueError("Please provide a staging folder.")
staging = sys.argv[sys.argv.index("--staging")+1]
tools.general.create_workflow_file_from_config(staging)
log.info("Workflow file created.")

def cli():
    step_map = {
        name.lower(): step
        for name, step in inspect.getmembers(steps)
        if inspect.isclass(step)
    }

    fire.Fire({**step_map, "all": All})
