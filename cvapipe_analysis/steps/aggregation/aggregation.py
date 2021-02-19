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

from .aggregation_tools import create_5d_hyperstacks

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
        df = pd.read_csv(path_to_shapemode_manifest, index_col='CellId')
        log.info(f"Shape of shape mode manifest: {df.shape}")

        # Merge the two dataframes (they do not have
        # necessarily the same size)
        df = df.merge(df_param[['CellRepresentationPath']], left_index=True, right_index=True)

        # Also read the manifest with paths to VTK files
        path_to_shapemode_paths = self.project_local_staging_dir / 'shapemode/shapemode.csv'
        if not path_to_shapemode_paths.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_shapemode_paths)
        df_shapemode_paths = pd.read_csv(path_to_shapemode_paths, index_col=0)
        log.info(f"Shape of shape mode paths manifest: {df_shapemode_paths.shape}")
        
        # Make necessary folders
        hyper_dir = self.step_local_staging_dir / 'hyperstacks'
        hyper_dir.mkdir(parents=True, exist_ok=True)

        # Agg representations per cells and shape mode.
        # Here we use principal components cerated with cell
        # nuclear SHE coefficients only (DNA_MEM_PCx).
        
        PREFIX = 'DNA_MEM_PC'
        
        pc_names = [f for f in df.columns if PREFIX in f]
        
        # Loop over shape modes
        df_hyperstacks_paths = pd.DataFrame([])
        for pc_idx, pc_name in enumerate(pc_names):

            log.info(f"Running PC: {pc_name}.")

            df_paths = create_5d_hyperstacks(
                df=df,
                df_paths=df_shapemode_paths,
                pc_names=pc_names,
                pc_idx=pc_idx,
                nbins=9,
                save_dir=hyper_dir
            )
            
            df_hyperstacks_paths = df_hyperstacks_paths.append(df_paths, ignore_index=True)
            
        # Save manifest
        self.manifest = df_hyperstacks_paths
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
