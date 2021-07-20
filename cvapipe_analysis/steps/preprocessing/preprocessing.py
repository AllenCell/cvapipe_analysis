#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

import pandas as pd
from cvapipe_analysis.tools import io, general
from .outliers_tools import outliers_removal
from .filtering_tools import filtering

log = logging.getLogger(__name__)

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
        filter = None,
        debug: bool=False,
        **kwargs
        ):
        
        with general.configuration(self.step_local_staging_dir) as control:
            
            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("computefeatures")
            log.info(f"Shape of manifest: {df.shape}")
            
            if control.remove_mitotics():

                if "cell_stage" not in df.columns:
                    raise ValueError("Column cell_stage not found.")
                df = df.loc[df.cell_stage=='M0']
                log.info(f"Manifest without mitotics: {df.shape}")
        
            if control.remove_outliers():

                path_to_outliers_folder = self.step_local_staging_dir/"outliers"
                path_to_outliers_folder.mkdir(parents=True, exist_ok=True)
                
                path_to_df_outliers = self.step_local_staging_dir/"outliers.csv"
                if not path_to_df_outliers.is_file() or control.overwrite():
                    log.info("Computing outliers...")
                    df_outliers = outliers_removal(df=df, output_dir=path_to_outliers_folder, log=log)
                    df_outliers.to_csv(path_to_df_outliers)
                else:
                    log.info("Using pre-detected outliers.")
                    df_outliers = pd.read_csv(path_to_df_outliers, index_col='CellId')

                df_outliers = df_outliers.loc[df.index]
                df.loc[df_outliers.index, 'Outlier'] = df_outliers['Outlier']
                df = df.loc[df.Outlier == 'No']
                df = df.drop(columns=['Outlier'])
                log.info(f"Shape of data without outliers: {df.shape}")
            
            if control.is_filtering_on():

                df = filtering(df, control)

            # Remove rows for which any feature is nan
            aliases = control.get_aliases_for_pca()
            columns = [f for f in df.columns if any(w in f for w in aliases)]
            columns = [c for c in columns if "transform" not in c]
            df_na = df.loc[df[columns].isna().any(axis=1)]
            if len(df_na):
                print(df_na.head())
                print(f"{len(df_na)} rows found with NaN values.")
                df = df.loc[~df.index.isin(df_na.index)]

            log.info(f"Saving manifest...")
            self.manifest = df
            manifest_path = self.step_local_staging_dir/'manifest.csv'
            self.manifest.to_csv(manifest_path)

            return manifest_path
