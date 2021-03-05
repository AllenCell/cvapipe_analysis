#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import NamedTuple, Optional, Union, List, Dict

import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from datastep import Step, log_run_params
from aics_dask_utils import DistributedHandler

from ...tools import general
from ...tools import cluster
from .compute_features_tools import load_images_and_calculate_features

log = logging.getLogger(__name__)

class SingleCellFeaturesResult(NamedTuple):
    CellId: Union[int, str]
    PathToFeatureJSON: Path


class SingleCellFeaturesError(NamedTuple):
    CellId: int
    error: str


class ComputeFeatures(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @staticmethod
    def _run_feature_extraction(
        row_index: int,
        row: pd.Series,
        save_dir: Path,
        load_data_dir: Path,
        overwrite: bool
    ) -> Union[SingleCellFeaturesResult, SingleCellFeaturesError]:

        # Get the ultimate end save path for this cell
        save_path = save_dir / f"{row_index}.json"

        # Check skip
        if not overwrite and save_path.is_file():
            log.info(f"Skipping cell feature generation for Cell Id: {row_index}")
            return SingleCellFeaturesResult(row_index, save_path)

        # Overwrite or didn't exist
        log.info(f"Beginning cell feature generation for CellId: {row_index}")

        channels = eval(row.name_dict)["crop_seg"]
        seg_path = load_data_dir / row.crop_seg
        
        # Wrap errors for debugging later
        try:
            load_images_and_calculate_features(
                path_seg=seg_path,
                channels=channels,
                path_output=save_path
            )
            log.info(f"Completed cell feature generation for CellId: {row_index}")
            return SingleCellFeaturesResult(row_index, save_path)

        # Catch and return error
        except Exception as e:
            log.info(
                f"Failed cell feature generation for CellId: {row_index}. Error: {e}"
            )
            return SingleCellFeaturesError(row_index, str(e))

    @staticmethod
    def _load_features_from_json(cell_feature_result):
        if cell_feature_result.PathToFeatureJSON.exists():
            with open(cell_feature_result.PathToFeatureJSON, "r") as fjson:
                features = json.load(fjson)
            return pd.Series(features, name=cell_feature_result.CellId)
        else:
            log.info(f"File not found: {str(cell_feature_result.PathToFeatureJSON)}.json")

    @log_run_params
    def run(
        self,
        debug=False,
        distributed_executor_address: Optional[str] = None,
        distribute: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ):

        # Load configuration file
        config = general.load_config_file()
        
        # Load manifest from previous step
        path_manifest = self.project_local_staging_dir / "loaddata/manifest.csv"
        df = pd.read_csv(path_manifest, index_col="CellId", low_memory=False)
        
        # Keep only the columns that will be used from now on
        columns_to_keep = ["crop_raw", "crop_seg", "name_dict"]
        df = df[columns_to_keep]
        
        # Create features directory
        features_dir = self.step_local_staging_dir / "cell_features"
        features_dir.mkdir(parents=True, exist_ok=True)

        load_data_dir = self.project_local_staging_dir / "loaddata"

        if distribute:
            
            nworkers = config['resources']['nworkers']
            data = cluster.data_to_distribute(df, nworkers)
            data.set_rel_path_to_dataframe(path_manifest)
            data.set_rel_path_to_input_images(load_data_dir)
            data.set_rel_path_to_output(features_dir)
            
            cluster.run_distributed_feature_extraction(data, config, log)

            log.info(f"{nworkers} have been launched. Please come back when the calculation is complete.")
            
            return None
            
        else:
            
            # Process each row
            with DistributedHandler(distributed_executor_address) as handler:
                # Start processing
                results = handler.batched_map(
                    self._run_feature_extraction,
                    *zip(*list(df.iterrows())),
                    [features_dir for i in range(len(df))],
                    [load_data_dir for i in range(len(df))],
                    [overwrite for i in range(len(df))],
                )

        # Generate features paths rows
        cell_features_dataset = []
        errors = []
        for result in results:
            if isinstance(result, SingleCellFeaturesResult):
                cell_features_dataset.append(result)
            else:
                errors.append(error)

        for error in errors:
            log.info(error)

        log.info("Reading features from JSON files. This might take a while.")

        N_CORES = len(os.sched_getaffinity(0))
        with concurrent.futures.ProcessPoolExecutor(N_CORES) as executor:
            df_features = list(tqdm(
                executor.map(self._load_features_from_json, cell_features_dataset),
                     total=len(cell_features_dataset)))
            
        df_features = pd.DataFrame(df_features)
        df_features.index = df_features.index.rename("CellId")
        
        log.info(f"Saving manifest of shape {df_features.shape}.")
        
        # Save manifest
        self.manifest = df_features
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        log.info("Manifest saved.")
        
        return manifest_save_path
