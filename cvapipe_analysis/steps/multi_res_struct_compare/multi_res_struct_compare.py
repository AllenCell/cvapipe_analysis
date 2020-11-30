#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from aics_dask_utils import DistributedHandler
from aicsimageio import AICSImage

from datastep import Step, log_run_params

from .utils import (
    compute_distance_metric,
    clean_up_results,
    make_plot,
    make_5d_stack_cross_corr_dataframe,
)

from .constants import (
    DatasetFieldsMorphed,
    DatasetFieldsIC,
    DatasetFieldsAverageMorphed,
)

from .exceptions import check_required_fields

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

REQUIRED_DATASET_FIELDS_MORPHED = [
    DatasetFieldsMorphed.StructureName1,
    DatasetFieldsMorphed.StructureName2,
    DatasetFieldsMorphed.SourceReadPath1,
    DatasetFieldsMorphed.SourceReadPath2,
    DatasetFieldsMorphed.Bin1,
    DatasetFieldsMorphed.Bin2,
    DatasetFieldsMorphed.CellId1,
    DatasetFieldsMorphed.CellId2,
]

REQUIRED_DATASET_FIELDS_IC = [
    DatasetFieldsIC.StructureName1,
    DatasetFieldsIC.StructureName2,
    DatasetFieldsIC.SourceReadPath1,
    DatasetFieldsIC.SourceReadPath2,
    DatasetFieldsIC.CellId,
    DatasetFieldsIC.SaveDir,
    DatasetFieldsIC.SaveRegPath,
]


class MultiResStructCompare(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        px_size=0.29,
        image_dims_crop_size=(64, 160, 96),
        input_csv_loc: Optional[str] = None,
        input_5d_stack: Optional[str] = None,
        max_rows: Optional[int] = None,
        permuted: Optional[bool] = None,
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        px_size: float
            How big are the (cubic) input pixels in micrometers
            Default: 0.29 (For IC cells)
            For morphed cells: 0.108

        image_dims_crop_size: Tuple[int]
            How to crop the input images before the resizing pyamid begins
            This is only for IC cells

        input_csv_loc: Optional[str]
            Path to input csv, can be a path to IC cells or morphed cells

            For IC cells, pass in the results of the generate_gfp_instantiations step
            Example:
            "/allen/aics/modeling/ritvik/projects/cvapipe/local_staging/
            /generategfpinstantiations/images_CellID_86655/manifest.csv"

            For morphed cells, pass in a csv containing paths to Matheus'
            morphed cells for a particular PC
            Example:
            "/allen/aics/modeling/ritvik/projects/cvapipe/
            FinalMorphedStereotypyDatasetPC1_9bins.csv"

            Note: Cannot provide this csv path along with 5d stack

        input_5d_stack: Optional[str]
            Path to the average morphed cell, this is a .tif
            Example:
            "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/
            assay-dev-cytoparam/avgcell/DNA_MEM_PC1_seg_avg.tif"

            Note: cannot provide this 5d stack along with a csv path

        permuted: Optional[bool]
            If True, compute permuted correlations at multi resolutions
            Default: None

        Returns
        -------
        result: pathlib.Path
            Path to manifest
        """

        # Adding hidden attributes to use in compute distance metric function
        self._input_csv_loc = input_csv_loc
        self._input_5d_stack = input_5d_stack
        Input_5d_Stack = None
        self._px_size = px_size

        # Make sure that both 5d stack and input csv are not provided at same time
        assert not all([self._input_csv_loc, self._input_5d_stack])

        # Handle dataset provided as string or path
        if isinstance(self._input_csv_loc, (str, Path)):
            dataset = Path(self._input_csv_loc).expanduser().resolve(strict=True)

            dataset = pd.read_csv(dataset)

            # Flag dataset as morphed cells
            if set(dataset.columns).intersection(set(REQUIRED_DATASET_FIELDS_IC)):
                REQUIRED_DATASET_FIELDS = REQUIRED_DATASET_FIELDS_IC
                # Check dataset and manifest have required fields
                check_required_fields(
                    dataset=dataset,
                    required_fields=REQUIRED_DATASET_FIELDS_IC,
                )

            # Flad dataset as IC cells
            if set(dataset.columns).intersection(set(REQUIRED_DATASET_FIELDS_MORPHED)):
                REQUIRED_DATASET_FIELDS = REQUIRED_DATASET_FIELDS_MORPHED
                # Check dataset and manifest have required fields
                check_required_fields(
                    dataset=dataset,
                    required_fields=REQUIRED_DATASET_FIELDS_MORPHED,
                )

            # Subset down
            dataset = dataset[REQUIRED_DATASET_FIELDS]

        # Handle dataset provided as string or path
        if isinstance(self._input_5d_stack, (str, Path)):
            Input_5d_Stack = (
                Path(self._input_5d_stack).expanduser().resolve(strict=True)
            )

            Input_5d_Stack = AICSImage(Input_5d_Stack).data

            assert Input_5d_Stack.shape[1] == DatasetFieldsAverageMorphed.NumStructs
            assert Input_5d_Stack.shape[3] == DatasetFieldsAverageMorphed.NumBins

            dataset = make_5d_stack_cross_corr_dataframe(
                DatasetFieldsAverageMorphed.NumStructs,
                DatasetFieldsAverageMorphed.NumBins,
            )

        if max_rows:
            dataset = dataset.head(max_rows)

        # Empty futures list
        distance_metric_futures = []
        distance_metric_results = []

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            distance_metric_future = handler.client.map(
                compute_distance_metric,
                [row for i, row in dataset.iterrows()],
                [Input_5d_Stack for i in range(len(dataset))],
                [permuted for i in range(len(dataset))],
                [px_size for i in range(len(dataset))],
                [image_dims_crop_size for i in range(len(dataset))],
            )

            distance_metric_futures.append(distance_metric_future)
            result = handler.gather(distance_metric_future)
            distance_metric_results.append(result)

        # Assemble final dataframe
        df_final = clean_up_results(distance_metric_results, permuted)

        # make a manifest
        self.manifest = pd.DataFrame(columns=["Description", "path"])

        # where to save outputs
        pairwise_dir = self.step_local_staging_dir / "pairwise_metrics"
        pairwise_dir.mkdir(parents=True, exist_ok=True)
        pairwise_loc = pairwise_dir / "multires_pairwise_similarity.csv"
        plot_dir = self.step_local_staging_dir / "pairwise_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_loc = plot_dir / "multi_resolution_image_correlation.png"

        # save pairwise dataframe to csv
        df_final.to_csv(pairwise_loc, index=False)
        self.manifest = self.manifest.append(
            {"Description": "raw similarity scores", "path": pairwise_loc},
            ignore_index=True,
        )

        # make a plot
        make_plot(data=df_final, plot_dir=plot_dir)
        self.manifest = self.manifest.append(
            {"Description": "plot of similarity vs resolution", "path": plot_loc},
            ignore_index=True,
        )

        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
