#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm

from cvapipe_analysis.tools import general, cluster, shapespace
from .compute_features_tools import FeatureCalculator

import pdb;
tr = pdb.set_trace

log = logging.getLogger(__name__)

class ComputeFeatures(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        distribute: Optional[bool]=False,
        overwrite: Optional[bool]=False,
        **kwargs
    ):
        
        # Load configuration file
        config = general.load_config_file()
        
        # Load parameterization dataframe
        path_to_loaddata_manifest = self.project_local_staging_dir / 'loaddata/manifest.csv'
        df = pd.read_csv(path_to_loaddata_manifest, index_col='CellId')
        log.info(f"Manifest: {df.shape}")
        
        # Make necessary folders
        save_dir = self.step_local_staging_dir / 'cell_features'
        save_dir.mkdir(parents=True, exist_ok=True)
                
        if distribute:
            nworkers = config['resources']['nworkers']            
            distributor = cluster.FeaturesDistributor(df, nworkers)
            distributor.distribute(config, log)
            log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")            
            return None
                
        calculator = FeatureCalculator(config)
        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            paths=list(executor.map(calculator.execute, [row for _,row in df.iterrows()]))
        df.loc[index,'PathToFeaturesFile'] = paths
        
        self.manifest = df
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
