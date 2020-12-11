#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict
from aicsfeature.extractor import cell, cell_nuc, dna
from scipy.ndimage import gaussian_filter as ndf

import aicsimageio
import dask.dataframe as dd
import pandas as pd
from aics_dask_utils import DistributedHandler
from aicsimageio import AICSImage
from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SingleCellFeaturesResult(NamedTuple):
    cell_id: Union[int, str]
    path: Path


class SingleCellFeaturesError(NamedTuple):
    cell_id: int
    error: str


class DatasetFields:
    CellId = "CellId"
    CellIndex = "CellIndex"
    FOVId = "FOVId"
    CellFeaturesPath = "CellFeaturesPath"


###############################################################################


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
        cell_ceiling_adjustment: int,
        save_dir: Path,
        overwrite: bool,
    ) -> Union[SingleCellFeaturesResult, SingleCellFeaturesError]:
        # Don't use dask for image reading
        aicsimageio.use_dask(False)

        # Get the ultimate end save path for this cell
        save_path = save_dir / f"{row.CellId}.json"

        # Check skip
        if not overwrite and save_path.is_file():
            log.info(f"Skipping cell feature generation for Cell Id: {row.CellId}")
            return SingleCellFeaturesResult(row.CellId, save_path)

        # Overwrite or didn't exist
        log.info(f"Beginning cell feature generation for CellId: {row.CellId}")

        # Wrap errors for debugging later
        try:
            raw_and_seg_images = []
            for i in ["crop_raw", "crop_seg"]:
                # image = AICSImage(row[f"{i}"])
                image = AICSImage(row[f"{i}"])

                # Preload image data
                image.data

                image = image.get_image_data("CYXZ", S=0, T=0)

                raw_and_seg_images.append(image)

            # Unpack channels
            nuc_seg = raw_and_seg_images[1][0]
            memb_seg = raw_and_seg_images[1][1]
            dna_image = raw_and_seg_images[0][0]
            memb_image = raw_and_seg_images[0][1]
            struct_image = raw_and_seg_images[0][2]

            # Adjust the DNA and membrane images
            adjusted_dna_image = dna_image.astype("uint16")
            adjusted_memb_image = memb_image.astype("uint16")

            # Simple deblur for better structure localization detection
            imf1 = ndf(struct_image, 5, mode="constant")
            imf2 = ndf(struct_image, 1, mode="constant")

            # Adjust structure image
            adjusted_struct_image = imf2 - imf1
            adjusted_struct_image[adjusted_struct_image < 0] = 0

            # Get features for the image using the adjusted images
            memb_nuc_struct_feats = cell_nuc.get_features(
                nuc_seg, memb_seg, adjusted_struct_image
            ).to_dict("records")[0]

            # Get DNA and membrane image features
            dna_feats = dna.get_features(adjusted_dna_image, seg=nuc_seg).to_dict(
                "records"
            )[0]
            memb_feats = cell.get_features(adjusted_memb_image, seg=memb_seg).to_dict(
                "records"
            )[0]

            # Combine all features
            features = {
                # **regularization_params,
                **dna_feats,
                **memb_feats,
                **memb_nuc_struct_feats,
            }

            # Save to JSON
            with open(save_path, "w") as write_out:
                json.dump(features, write_out)

            log.info(f"Completed cell feature generation for CellId: {row.CellId}")
            return SingleCellFeaturesResult(row.CellId, save_path)

        # Catch and return error
        except Exception as e:
            log.info(
                f"Failed cell feature generation for CellId: {row.CellId}. Error: {e}"
            )
            return SingleCellFeaturesError(row.CellId, str(e))

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
        cell_ceiling_adjustment: int = 0,
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Provided a dataset generate a features JSON file for each cell.
        Parameters
        ----------
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame]
            The primary cell dataset to use for generating features JSON for each cell.
            **Required dataset columns:** *["CellId", "CellIndex", "FOVId",
            "StandardizedFOVPath"]*
        cell_ceiling_adjustment: int
            The adjust to use for raising the cell shape ceiling. If <= 0, this will be
            ignored and cell data will be selected but not adjusted.
            Default: 0
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
            Default: None
        batch_size: Optional[int]
            An optional batch size to process n features at a time.
            Default: None (Process all at once)
        overwrite: bool
            If this step has already partially or completely run, should it overwrite
            the previous files or not.
            Default: False (Do not overwrite or regenerate files)
        Returns
        -------
        manifest_save_path: Path
            Path to the produced manifest with the CellFeaturesPath column added.
        """
        # Handle dataset provided as string or path
        if isinstance(dataset, (str, Path)):
            dataset = Path(dataset).expanduser().resolve(strict=True)

            # Read dataset
            dataset = pd.read_csv(dataset)

            dataset = dataset.head(5)

        # Create features directory
        features_dir = self.step_local_staging_dir / "cell_features"
        features_dir.mkdir(exist_ok=True)

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            results = handler.batched_map(
                self._generate_single_cell_features,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(dataset.iterrows())),
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [cell_ceiling_adjustment for i in range(len(dataset))],
                [features_dir for i in range(len(dataset))],
                [overwrite for i in range(len(dataset))],
                batch_size=batch_size,
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

        # Convert features paths rows to dataframe
        cell_features_dataset = pd.DataFrame(cell_features_dataset)

        # Drop CellFeaturesPath column if it already exists
        if DatasetFields.CellFeaturesPath in dataset.columns:
            dataset = dataset.drop(columns=[DatasetFields.CellFeaturesPath])

        # Join original dataset to the fov paths
        self.manifest = dataset.merge(cell_features_dataset, on=DatasetFields.CellId)

        # Save manifest to CSV
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path, index=False)

        # Save errored cells to JSON
        with open(self.step_local_staging_dir / "errors.json", "w") as write_out:
            json.dump(errors, write_out)

        return manifest_save_path
