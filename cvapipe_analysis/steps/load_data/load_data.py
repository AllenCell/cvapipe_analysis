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
    package_name="aics/hipsc_single_cell_image_dataset",
    registry="s3://allencell",
    data_save_loc="quilt_data",
    ignore_warnings=True,
    test=False,
):
    """Download a quilt dataset and supress nfs file attribe warnings by default"""
    pkg = quilt3.Package.browse(package_name, registry)
    
    meta_df = pkg["metadata.csv"]()
    
    if test:
        print('>> Downloading test dataset...')
        meta_df = meta_df.sample(n=300, random_state=666)

    # Creating directories
    path_to_raw_folder = (data_save_loc / Path("crop_raw"))
    path_to_raw_folder.mkdir(parents=True, exist_ok=True)
    path_to_seg_folder = (data_save_loc / Path("crop_seg"))
    path_to_seg_folder.mkdir(parents=True, exist_ok=True)
        
    for i, row in meta_df.iterrows():
        if ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # download single cell raw and segmentation images
                # (cell membrane dye, dna dye, structure)
                pkg[row["crop_raw"]].fetch(path_to_raw_folder)
                pkg[row["crop_seg"]].fetch(path_to_seg_folder)
        else:
            # download single cell raw and segmentation images
            # (cell membrane dye, dna dye, structure)
            pkg[row["crop_raw"]].fetch(path_to_raw_folder)
            pkg[row["crop_seg"]].fetch(path_to_seg_folder)

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
        package_name="aics/hipsc_single_cell_image_dataset",
        registry="s3://allencell",
        data_save_loc="quilt_data",
        test=False,
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
        test: bool
            Download only a small test dataset of 300 cells chosen at random.

        Parameters
        ----------

        Returns
        -------
        result: Any
            A pickable object or value that is the result of any processing you do.
        """
        dataset = download_quilt_data(
            package_name=package_name,
            registry=registry,
            data_save_loc=self.step_local_staging_dir,
            ignore_warnings=True,
            test=test
        )
        
        self.manifest = dataset

        # Save manifest to CSV
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path, index=False)

        return manifest_save_path
