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

        # For parameterization we need to load the single cell
        # metadata dataframe and the single cell feature dataframe

        # Load manifest from load_data step
        df = pd.read_csv(
            self.project_local_staging_dir/'loaddata/manifest.csv',
            index_col = 'CellId'
        )
        
        # Keep only the columns that will be used from now on
        columns_to_keep = ['crop_raw', 'crop_seg', 'name_dict']
        df = df[columns_to_keep]
        
        # Load manifest from feature calculation step
        df_features = pd.read_csv(
            self.project_local_staging_dir/'computefeatures/manifest.csv',
            index_col = 'CellId'
        )
                
        # Merge the two dataframes
        df = df.join(df_features, how='inner')
        
        self.manifest = df
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)
            
        return manifest_save_path
