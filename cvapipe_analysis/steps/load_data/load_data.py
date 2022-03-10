#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

from cvapipe_analysis.tools import general
from .load_data_tools import DataLoader

log = logging.getLogger(__name__)

class LoadData(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(self, **kwargs):

        with general.configuration(self.step_local_staging_dir) as control:
            manifest_path = self.step_local_staging_dir / 'manifest.csv'
        
            if control.overwrite() or not os.path.isfile(manifest_path):
                loader = DataLoader(control)
                df = loader.load(kwargs)
                self.manifest = df
                self.manifest.to_csv(manifest_path)
            else:
                print(f"Skipping LoadData, manifest already exists at {manifest_path}")

        return manifest_path