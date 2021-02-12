#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
import quilt3

import pandas as pd
from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def download_quilt_data(
    package_name = "aics/hipsc_single_cell_image_dataset",
    registry = "s3://allencell",
    data_save_loc = "quilt_data",
    ignore_warnings = True,
    test = False,
):
    """Download a quilt dataset and supress nfs file attribe warnings by default"""
    pkg = quilt3.Package.browse(package_name, registry)
    
    df_meta = pkg["metadata.csv"]()
    
    if test:
        df_test = pd.DataFrame([])
        print('>> Downloading test dataset with 12 interphase cell images per structure.')
        df_meta = df_meta.loc[df_meta.cell_stage=='M0']
        for struct, df_struct in df_meta.groupby('structure_name'):
            df_test = df_test.append(df_struct.sample(n=12, random_state=666, replace=False).copy())
        df_meta = df_test.copy()

    for i, row in df_meta.iterrows():
        if ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # download single cell raw and segmentation images
                # (cell membrane dye, dna dye, structure)
                pkg[row["crop_raw"]].fetch(data_save_loc/row["crop_raw"])
                pkg[row["crop_seg"]].fetch(data_save_loc/row["crop_seg"])
        else:
            # download single cell raw and segmentation images
            # (cell membrane dye, dna dye, structure)
            pkg[row["crop_raw"]].fetch(data_save_loc/row["crop_raw"])
            pkg[row["crop_seg"]].fetch(data_save_loc/row["crop_seg"])

    return df_meta

class LoadData(Step):

    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        package_name = "aics/hipsc_single_cell_image_dataset",
        registry = "s3://allencell",
        data_save_loc = "quilt_data",
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
            Download only a small test dataset of 300 interphase cells chosen at
            random (12 cells per structure).

        Parameters
        ----------

        Returns
        -------
        result: Any
            A pickable object or value that is the result of any processing you do.
        """
        dataset = download_quilt_data(
            package_name = package_name,
            registry = registry,
            data_save_loc = self.step_local_staging_dir,
            ignore_warnings = True,
            test = test
        )
        
        self.manifest = dataset

        # Save manifest to CSV
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path, index=False)

        return manifest_save_path
