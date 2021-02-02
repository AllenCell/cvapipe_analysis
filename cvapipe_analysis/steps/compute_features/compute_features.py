#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

import pandas as pd
from tqdm import tqdm
from datastep import Step, log_run_params
from aics_dask_utils import DistributedHandler
from .compute_features_tools import get_segmentations, get_features
import numpy as np    

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


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
    def _generate_single_cell_features(
        row_index: int,
        row: pd.Series,
        save_dir: Path,
        load_data_dir: Path,
        overwrite: bool,
    ) -> Union[SingleCellFeaturesResult, SingleCellFeaturesError]:

        # Get the ultimate end save path for this cell
        save_path = save_dir / f"{row_index}.json"

        # Check skip
        if not overwrite and save_path.is_file():
            log.info(f"Skipping cell feature generation for Cell Id: {row_index}")
            return SingleCellFeaturesResult(row_index, save_path)

        # Overwrite or didn't exist
        log.info(f"Beginning cell feature generation for CellId: {row_index}")

        # Wrap errors for debugging later
        try:
            # Find the correct segmentation for nucleus,
            # cell and structure
            # channels = df.at[index,'name_dict']
            channels = row.name_dict
            seg_dna, seg_mem, seg_str = get_segmentations(
                folder=load_data_dir,
                path_to_seg=row.crop_seg,
                channels=eval(channels)['crop_seg']
            )

            # Compute nuclear features
            features_dna = get_features(
                input_image=seg_dna,
                input_reference_image=seg_mem
            )

            # Compute cell features
            features_mem = get_features(
                input_image=seg_mem,
                input_reference_image=seg_mem
            )

            # Compute structure features
            features_str = get_features(
                input_image=seg_str,
                input_reference_image=None,
                compute_shcoeffs=False
            )

            # Append prefix to features names
            features_dna = dict(
                (f'dna_{key}', value) for (key, value) in features_dna.items()
            )
            features_mem = dict(
                (f'mem_{key}', value) for (key, value) in features_mem.items()
            )
            features_str = dict(
                (f'str_{key}', value) for (key, value) in features_str.items()
            )    

            # Concatenate all features for this cell
            features = features_dna.copy()
            features.update(features_mem)
            features.update(features_str)

            # Save to JSON
            with open(save_path, "w") as write_out:
                json.dump(features, write_out)

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
        overwrite: bool = False,
        **kwargs
    ):

        # Load manifest from previous step
        df = pd.read_csv(
            self.project_local_staging_dir / 'loaddata/manifest.csv',
            index_col='CellId'
        )
        df = pd.read_csv('./test.csv')

        # Keep only the columns that will be used from now on
        columns_to_keep = ['crop_raw', 'crop_seg', 'name_dict']
        df = df[columns_to_keep]
      
        # Sample the dataset if running debug mode
        if debug:
            df = df.sample(n=8, random_state=666)

        # Create features directory
        features_dir = self.step_local_staging_dir / "cell_features"
        features_dir.mkdir(exist_ok=True)

        load_data_dir = self.project_local_staging_dir / 'loaddata'

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            results = handler.batched_map(
                self._generate_single_cell_features,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(df.iterrows())),
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [features_dir for i in range(len(df))],
                [load_data_dir for i in range(len(df))],
                [overwrite for i in range(len(df))]
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

        # Gather all features into a single manifest
        df_features = pd.DataFrame([])
        for index in tqdm(df.index, desc='Merging features'):
            with open(self.step_local_staging_dir / f"cell_features/{index}.json", "r") as fjson:
                features = json.load(fjson)
                features = pd.Series(features, name=index)
                df_features = df_features.append(features)
        df_features.index = df_features.index.rename('CellId')

        # Save manifest
        self.manifest = df_features
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)
            
        return manifest_save_path
