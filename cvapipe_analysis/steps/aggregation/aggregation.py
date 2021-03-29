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

from cvapipe_analysis.tools import general, cluster, shapespace
from .aggregation_tools import Aggregator

import pdb;
tr = pdb.set_trace

log = logging.getLogger(__name__)

class Aggregation(Step):
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
        
        with general.configuration(self.step_local_staging_dir) as config:
        
            # Load parameterization dataframe
            path_to_param_manifest = self.project_local_staging_dir / 'parameterization/manifest.csv'
            if not path_to_param_manifest.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_param_manifest)
            df_param = pd.read_csv(path_to_param_manifest, index_col='CellId')
            log.info(f"Shape of param manifest: {df_param.shape}")

            # Load shape modes dataframe
            path_to_shapemode_manifest = self.project_local_staging_dir / 'shapemode/manifest.csv'
            if not path_to_shapemode_manifest.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_shapemode_manifest)
            df = pd.read_csv(path_to_shapemode_manifest, index_col='CellId', low_memory=False)
            log.info(f"Shape of shape mode manifest: {df.shape}")

            # Merge the two dataframes (they do not have
            # necessarily the same size)
            df = df.merge(df_param[['PathToRepresentationFile']], left_index=True, right_index=True)

            # Make necessary folders
            for folder in ['repsagg','aggmorph']:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            # Create all combinations of parameters
            df_agg = general.create_agg_dataframe_of_celids(df, config)

            if distribute:

                nworkers = config['resources']['nworkers']            
                distributor = cluster.AggregationDistributor(df_agg, nworkers)
                distributor.distribute(config, log)

                log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")            
                return None

            space = shapespace.ShapeSpaceBasic(config)
            aggregator = Aggregator(config)
            aggregator.set_shape_space(space)
            for index, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                '''Concurrent processes inside. Do not use concurrent here.'''
                df_agg.loc[index,'PathToAggFile'] = aggregator.execute(row)

            log.info(f"Saving manifest...")
            self.manifest = df_agg
            manifest_path = self.step_local_staging_dir / 'manifest.csv'
            self.manifest.to_csv(manifest_path)

        return manifest_path
