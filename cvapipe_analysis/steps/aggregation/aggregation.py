#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import pandas as pd
from tqdm import tqdm

from .aggregation_tools import aggregate_intensities_of_shape_mode

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Aggregation(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        overwrite: bool = False,
        **kwargs
    ):

        # Load parameterization dataframe
        path_to_param_manifest = self.project_local_staging_dir / 'parameterization/manifest.csv'
        if not path_to_param_manifest.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_param_manifest)
        df_param = pd.read_csv(path_to_param_manifest, index_col='CellId')
        log.info(f"Shape of param manifest: {df_param.shape}")

        # Load shape modes dataframe
        path_to_shapemode_manifest = self.project_local_staging_dir / 'shapemode/manifest.csv'
        if not path_to_shapemode_manifest.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_shapemode_manifest)
        df_shapemode = pd.read_csv(path_to_shapemode_manifest, index_col='CellId')
        log.info(f"Shape of shape mode manifest: {df_shapemode.shape}")

        # Merge the two dataframes (they do not have
        # necessarily the same size)
        df = df_shapemode.merge(df_param[['CellRepresentationPath']], left_index=True, right_index=True)
                
        # Make necessary folders
        agg_dir = self.step_local_staging_dir / 'agg_representations'
        agg_dir.mkdir(parents=True, exist_ok=True)

        # Agg representations per cells and shape mode.
        # Here we use principal components cerated with cell
        # nuclear SHE coefficients only (DNA_MEM_PCx).
        
        PREFIX = 'DNA_MEM_PC'
        
        pc_names = [f for f in df.columns if PREFIX in f]
        
        # Loop over shape modes
        df_agg = pd.DataFrame([])
        for pc_idx, pc_name in enumerate(pc_names):

            result = aggregate_intensities_of_shape_mode(
                df = df,
                pc_names = pc_names,
                pc_idx = pc_idx,
                save_dir = agg_dir
            )
            
            df_agg = df_agg.append(pd.DataFrame(result), ignore_index=True)

        # Save manifest
        self.manifest = df_agg
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
