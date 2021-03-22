#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm

from cvapipe_analysis.tools import general, cluster, shapespace
from .shapemode_tools import ShapeModeCalculator

from .outliers import outliers_removal
from .dim_reduction import pca_analysis
from .avgshape import digitize_shape_mode
from .avgshape import get_shcoeffs_from_pc_coords
from .avgshape import animate_shape_modes_and_save_meshes
from .plotting import paired_correlation, dataset_summary_table

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class Shapemode(Step):
    
    def __init__(
        self,
        direct_upstream_tasks: List['Step'] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        debug: bool = False,
        overwrite: bool = False,
        remove_mitotics: bool = True,
        no_structural_outliers: bool = False,
        **kwargs
    ):
        
        # Load configuration file
        config = general.load_config_file()
        
        # Load parameterization dataframe
        path_to_loaddata_manifest = self.project_local_staging_dir / 'computefeatures/manifest.csv'
        df = pd.read_csv(path_to_loaddata_manifest, index_col='CellId')
        log.info(f"Manifest: {df.shape}")
        
        # Make necessary folders
        for folder in ['pca', 'avgshape']:
            save_dir = self.step_local_staging_dir / folder
            save_dir.mkdir(parents=True, exist_ok=True)
                        
        calculator = ShapeModeCalculator(config)
        calculator.workflow(df)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #####################################
        
        # Load feature dataframe
        path_to_features_manifest = self.project_local_staging_dir / 'computefeatures/manifest.csv'
        df = pd.read_csv(path_to_features_manifest, index_col='CellId')
        log.info(f"Shape of features data: {df.shape}")

        # Perform principal component analysis
        dir_pca = self.step_local_staging_dir / 'pca'
        dir_pca.mkdir(parents=True, exist_ok=True)
        dir_avgshape = self.step_local_staging_dir / 'avgshape'
        dir_avgshape.mkdir(parents=True, exist_ok=True)
        
        aliases = config['pca']['aliases']
        prefix = "_".join(aliases)
        feature_prefixes = [f"{alias}_shcoeffs_L" for alias in aliases]

        log.info(f"[{prefix}] - PCA on {', '.join(feature_prefixes)}")

        features = [
            fn for fn in df.columns if any(word in fn for word in feature_prefixes)
        ]

        # PCA
        df, pc_names, pca = pca_analysis(
            df = df,
            feature_names = features,
            prefix = prefix,
            npcs_to_calc = config['pca']['number_of_pcs'],
            save = dir_pca / f'pca_{prefix}'
        )

        # Make plot of PC cross correlations
        paired_correlation(
            df = df,
            features = pc_names,
            save = dir_pca / f'correlations_{prefix}'
        )

        import pdb; pdb.set_trace()
        
        # Shape modes

        # Calculates the average shapes along each PC
        for mode, pc_name in enumerate(pc_names):

            log.info(
                f"\tCalculating average cell and nuclear shape for mode: {pc_name}"
            )

            # Create map points, bins e get all cells in each bin
            df_filtered, bin_indexes, (bin_centers, pc_std) = digitize_shape_mode(
                df = df,
                feature = pc_name,
                nbins = config['pca']['number_map_points'],
                filter_based_on = pc_names,
                save = dir_avgshape / pc_name,
            )

            # Convert map points back to SH coefficients
            df_mappoints_shcoeffs = get_shcoeffs_from_pc_coords(
                coords = bin_centers * pc_std,
                pc = mode,
                pca = pca
            )

            if 'DNA' not in pc_name:
                # Create fake DNA coeffs for viz purposes if shape
                # space was generated with cell alone
                df_tmp = df_mappoints_shcoeffs.copy()
                df_tmp.columns = [f.replace('mem', 'dna') for f in df_tmp.columns]
                df_mappoints_shcoeffs = pd.concat(
                    [df_mappoints_shcoeffs, df_tmp], axis=1
                )

            if 'MEM' not in pc_name:
                # Create fake cell coeffs for viz purposes if shape
                # space was generated with nucleus alone
                df_tmp = df_mappoints_shcoeffs.copy()
                df_tmp.columns = [f.replace('dna', 'mem') for f in df_tmp.columns]
                df_mappoints_shcoeffs = pd.concat(
                    [df_mappoints_shcoeffs, df_tmp], axis=1
                )

            # Reconstruct cell and nuclear shape. Also adjust nuclear
            # position relative to the cell when the shape space is
            # created by a joint combination of cell and nuclear shape.
            df_paths = animate_shape_modes_and_save_meshes(
                df_agg = df_mappoints_shcoeffs,
                mode = pc_name,
                save = dir_avgshape,
                fix_nuclear_position = None if prefix != 'DNA_MEM' else (df_filtered,bin_indexes),
                plot_limits = config['pca']['plot_limits'],
            )

            df_shapemode_paths = df_shapemode_paths.append(df_paths, ignore_index=True)

        # Save datafrme with path for meshes and gifs
        dataframe_path = self.step_local_staging_dir / 'shapemode.csv'
        df_shapemode_paths.to_csv(dataframe_path)
        # Save manifest
        self.manifest = df
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
