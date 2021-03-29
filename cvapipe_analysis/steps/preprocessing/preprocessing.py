#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm

from cvapipe_analysis.tools import general
from .preprocessing_tools import outliers_removal

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class Preprocessing(Step):
    
    def __init__(
        self,
        direct_upstream_tasks: List['Step'] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        debug: bool = False,
        overwrite: bool = False,
        **kwargs
    ):
        
        with general.configuration(self.step_local_staging_dir) as config:
            
            # Load metadata dataframe
            path_to_metadata_manifest = self.project_local_staging_dir / 'loaddata/manifest.csv'
            df_meta = pd.read_csv(path_to_metadata_manifest, index_col='CellId')
            # Drop unwanted columns
            df_meta = df_meta[            
                [c for c in df_meta.columns if not any(s in c for s in ['mem_','dna_','str_'])]
            ]
            log.info(f"Shape of metadata: {df_meta.shape}")

            # Load feature dataframe
            path_to_features_manifest = self.project_local_staging_dir / 'computefeatures/manifest.csv'
            df_features = pd.read_csv(path_to_features_manifest, index_col='CellId')
            log.info(f"Shape of features data: {df_features.shape}")

            # Merged dataframe
            df = pd.concat([df_meta,df_features], axis=1)
            log.info(f"Manifest: {df.shape}")
            
            if config["preprocessing"]["remove_mitotics"]:
                
                if "cell_stage" not in df.columns:
                    raise ValueError("Column cell_stage not found.")
                df = df.loc[df.cell_stage=='M0']
                log.info(f"Manifest without mitotics: {df.shape}")
        
            if config["preprocessing"]["remove_outliers"]:

                path_to_outliers_folder = self.step_local_staging_dir / "outliers"
                path_to_outliers_folder.mkdir(parents=True, exist_ok=True)
                
                path_to_df_outliers = self.step_local_staging_dir/"outliers.csv"
                if not config["project"]["overwrite"] or path_to_df_outliers.is_file():
                    log.info("Using pre-detected outliers.")
                    df_outliers = pd.read_csv(path_to_df_outliers, index_col='CellId')
                else:
                    log.info("Computing outliers...")
                    df_outliers = outliers_removal(df=df, output_dir=path_to_outliers_folder, log=log)
                    df_outliers.to_csv(manifest_outliers_path)
                df.loc[df_outliers.index, 'Outlier'] = df_outliers['Outlier']
                df = df.loc[df.Outlier == 'No']
                df = df.drop(columns=['Outlier'])
                log.info(f"Shape of data without outliers: {df.shape}")
            
            self.manifest = df
            manifest_path = self.step_local_staging_dir / 'manifest.csv'
            self.manifest.to_csv(manifest_path)
            return manifest_path
