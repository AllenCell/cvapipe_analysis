#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import NamedTuple, Optional, Union, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from datastep import Step, log_run_params
from aics_dask_utils import DistributedHandler

from ...tools import general
from ...tools import cluster
from .compute_features_tools import load_images_and_calculate_features

log = logging.getLogger(__name__)


class DatasetFields:
    CellId = "CellId"
    CellIndex = "CellIndex"
    FOVId = "FOVId"
    CellFeaturesPath = "CellFeaturesPath"


class SingleCellFeaturesResult(NamedTuple):
    cell_id: Union[int, str]
    path: Path


class SingleCellFeaturesError(NamedTuple):
    cell_id: int
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
        df = pd.read_csv(path_manifest, index_col="CellId")
        
        # Keep only the columns that will be used from now on
        columns_to_keep = ["crop_raw", "crop_seg", "name_dict"]
        df = df[columns_to_keep]
        
        # Create features directory
        features_dir = self.step_local_staging_dir / "cell_features"
        features_dir.mkdir(parents=True, exist_ok=True)

        load_data_dir = self.project_local_staging_dir / "loaddata"

        if distribute:
            
            cluster.run_distributed_feature_extraction(
                df,
                path_manifest,
                load_data_dir,
                features_dir,
                config,
                log)

            log.info(f"{config['resources']['nworkers']} have been launched. Please come back when the calculation is complete.")
            
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
                cell_features_dataset.append(
                    {
                        DatasetFields.CellId: result.cell_id,
                        DatasetFields.CellFeaturesPath: result.path,
                    }
                )
            else:
                errors.append(
                    {DatasetFields.CellId: result.cell_id, "Error": result.error}
                )

        for error in errors:
            log.info(error)

        # Gather all features into a single manifest
        df_features = pd.DataFrame([])
        for index in tqdm(df.index, desc="Merging features"):
            if (self.step_local_staging_dir / f"cell_features/{index}.json").exists():
                with open(self.step_local_staging_dir / f"cell_features/{index}.json", "r") as fjson:
                    features = json.load(fjson)
                features = pd.Series(features, name=index)
                df_features = df_features.append(features)
            else:
                log.info(f"File not found: {index}.json")
                
        df_features.index = df_features.index.rename("CellId")

        # Save manifest
        self.manifest = df_features
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
