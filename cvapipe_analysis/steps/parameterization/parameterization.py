#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

import pandas as pd
from tqdm import tqdm
from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class Parameterization(Step):

    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        debug=False,
        **kwargs):

        self.manifest = df_features
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)
            
        return manifest_save_path
