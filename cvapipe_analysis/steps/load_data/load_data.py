#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
import quilt3

from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def download_quilt_data(
    package="rorydm/mitotic_annotations",
    registry="s3://allencell-internal-quilt",
    data_save_loc="quilt_data",
    ignore_warnings=True,
):
    """download a quilt dataset and supress nfs file attribe warnings by default"""
    dataset_manifest = quilt3.Package.browse(package, registry)

    meta_df = dataset_manifest["metadata.csv"]()

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # download single cell raw images (cell membrane dye, dna dye, structure)
            dataset_manifest["crop_raw"].fetch(data_save_loc / Path("crop_raw"))

            # download single cell segmentation images (cell seg, nucleus seg,
            # and structure seg)
            dataset_manifest["crop_seg"].fetch(data_save_loc / Path("crop_seg"))

    else:
        # download single cell raw images (cell membrane dye, dna dye, structure)
        dataset_manifest["crop_raw"].fetch(data_save_loc / Path("crop_raw"))

        # download single cell segmentation images (cell seg, nucleus seg,
        #  and structure seg)
        dataset_manifest["crop_seg"].fetch(data_save_loc / Path("crop_seg"))

    return meta_df


class LoadData(Step):
    # We may want to vary:
    # * The label we are attaching to each image
    # * which 2d images we are using as input

    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        package="aics/hipsc_single_cell_image_dataset",
        registry="s3://allencell",
        data_save_loc="quilt_data",
        **kwargs
    ):
        """
        Run a pure function.

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------

        Returns
        -------
        result: Any
            A pickable object or value that is the result of any processing you do.
        """
        dataset = download_quilt_data(
            package=package,
            registry=registry,
            data_save_loc=self.step_local_staging_dir,
            ignore_warnings=True,
        )

        self.manifest = dataset

        # Save manifest to CSV
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path, index=False)

        return manifest_save_path
