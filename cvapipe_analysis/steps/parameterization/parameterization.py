#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import warnings
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

import pandas as pd
from tqdm import tqdm
from datastep import Step, log_run_params
from aics_dask_utils import DistributedHandler

from cvapipe_analysis.tools import general, cluster
from .parameterization_tools import Parameterizer

log = logging.getLogger(__name__)

class SingleCellParameterizationResult(NamedTuple):
    CellId: Union[int, str]
    PathToRepresentationFile: Path

class SingleCellParameterizationError(NamedTuple):
    CellId: int
    error: str

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
        distributed_executor_address: Optional[str] = None,
        distribute: Optional[bool] = False,
        overwrite: bool = False,
        **kwargs):

        # Load configuration file
        config = general.load_config_file()
        
        # For parameterization we need to load the single cell
        # metadata dataframe and the single cell features dataframe

        # Load manifest from load_data step
        path_meta_manifest = self.project_local_staging_dir/'loaddata/manifest.csv'
        df = pd.read_csv(path_meta_manifest, index_col='CellId', low_memory=False)
        
        # Keep only the columns that will be used from now on
        columns_to_keep = ['structure_name','crop_raw', 'crop_seg', 'name_dict']
        df = df[columns_to_keep]
        
        # Load manifest from feature calculation step
        path_features_manifest = self.project_local_staging_dir/'computefeatures/manifest.csv'
        df_features = pd.read_csv(path_features_manifest, index_col='CellId', low_memory=False)
                
        # Merge the two dataframes
        df = df.join(df_features, how='inner')
        
        # Folder for storing the parameterized intensity representations
        save_dir = self.step_local_staging_dir/'representations'
        save_dir.mkdir(parents=True, exist_ok=True)
        df['PathToOutputFolder'] = str(save_dir)
        
        # Data folder
        load_data_dir = self.project_local_staging_dir/'loaddata'
        
        if distribute:
            
            nworkers = config['resources']['nworkers']            
            distributor = cluster.ParameterizationDistributor(df, nworkers)
            distributor.distribute(config, log)

            log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")            
            return None

        else:
            

        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            PathToRepresentationFiles=list(
                executor.map(parameterizer.execute, [row for _,row in df.iterrows()])
            )
        df.loc[index,'PathToRepresentationFile'] = PathToRepresentationFiles

        self.manifest = df[['PathToRepresentationFile']]
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path

