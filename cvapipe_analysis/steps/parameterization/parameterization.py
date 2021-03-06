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

from cvapipe_analysis.tools import general
from cvapipe_analysis.tools import cluster
from .parameterization_tools import parameterize

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

    @staticmethod
    def _single_cell_parameterization(
        index: int,
        row: pd.Series,
        save_dir: Path,
        load_data_dir: Path,
        overwrite: bool,
    ) -> Union[SingleCellParameterizationResult, SingleCellParameterizationError]:

        # Get the ultimate end save path for this cell
        save_path = save_dir / f"{index}.tif"

        if not overwrite and save_path.is_file():
            log.info(f"Skipping cell parameterization for Cell Id: {index}")
            return SingleCellParameterizationResult(index, save_path)

        log.info(f"Beginning cell parameterization for CellId: {index}")

        try:
            parameterize(load_data_dir, row, save_path)            
            log.info(f"Completed cell parameterization for CellId: {index}")
            return SingleCellParameterizationResult(index, save_path)

        except Exception as e:
            log.info(f"Failed cell parameterization for CellId: {index}. Error: {e}")
            return SingleCellParameterizationError(index, str(e))

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

        # Data folder
        load_data_dir = self.project_local_staging_dir/'loaddata'
        
        if distribute:
            
            log.info(f"Saving dataframe for workers...")
            path_manifest = Path(".distribute/manifest.csv")
            df.to_csv(path_manifest)
            
            nworkers = config['resources']['nworkers']
            data = cluster.data_to_distribute(df, nworkers)
            data.set_rel_path_to_dataframe(path_manifest)
            data.set_rel_path_to_input_images(load_data_dir)
            data.set_rel_path_to_output(save_dir)
            python_file = "cvapipe_analysis/steps/parameterization/parameterization_tools.py"
            cluster.distribute_python_code(data, config, log, python_file)

            log.info(f"{nworkers} have been launched. Please come back when the calculation is complete.")
            
            return None

        else:
            
            with DistributedHandler(distributed_executor_address) as handler:
                results = handler.batched_map(
                    self._single_cell_parameterization,
                    *zip(*list(df.iterrows())),
                    [save_dir for i in range(len(df))],
                    [load_data_dir for i in range(len(df))],
                    [overwrite for i in range(len(df))]
                )

        # Generate features paths rows
        errors = []
        df_param = []
        for result in results:
            if isinstance(result, SingleCellParameterizationResult):
                df_param.append(result)
            else:
                errors.append(result)
        # Convert to DataFrame
        df_param = pd.DataFrame(df_param).set_index('CellId')

        # Display errors if any
        if len(errors)>0:
            warnings.warn("One or more errors found.")
            print(errors)
                
        # Save manifest
        self.manifest = df_param
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
