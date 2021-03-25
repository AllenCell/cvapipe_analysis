#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import numpy as np
import pandas as pd
from tqdm import tqdm

from cvapipe_analysis.tools import general, cluster, shapespace, plotting
from .load_data_tools import DataLoader

import pdb;
tr = pdb.set_trace

log = logging.getLogger(__name__)

class LoadData(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        #test: Optional[bool]=False,
        #csv: Optional[Path]=None,
        **kwargs
    ):

        # Load configuration file
        config = general.load_config_file()
        
        loader = DataLoader(config)
        df = loader.load(kwargs)
        
        self.manifest = df
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)
        
        return manifest_path
