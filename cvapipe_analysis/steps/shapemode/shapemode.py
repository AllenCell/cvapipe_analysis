#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import pandas as pd

from ...tools import general
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
        
        # Load feature dataframe
        path_to_metadata_manifest = self.project_local_staging_dir / 'loaddata/manifest.csv'
        df_meta = pd.read_csv(path_to_metadata_manifest, index_col='CellId')
        # Drop unwanted columns
        df_meta = df_meta[            
            [c for c in df_meta.columns if not any(s in c for s in ['mem_','dna_','str_'])]
        ]
        log.info(f"Shape of metadata: {df_meta.shape}")
        
        # Load feature dataframe
        path_to_features_manifest = self.project_local_staging_dir / 'computefeatures/manifest.csv'
        df_features = pd.read_csv(path_to_features_manifest, index_col='CellId')
        log.info(f"Shape of features data: {df_features.shape}")
        
        # Make necessary folders
        tables_dir = self.step_local_staging_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)

        # Perform outlier detection based on Gaussian kernel estimation
        outliers_dir = self.step_local_staging_dir / 'outliers'
        outliers_dir.mkdir(exist_ok=True)

        # Mitotic removal
        
        if 'cell_stage' in df_meta.columns:
        
            # Generate table with number of mitotic vs. intephase cells
            dataset_summary_table(
                df = df_meta,
                levels = ['cell_stage'],
                factor = 'structure_name',
                save = tables_dir / 'cell_stage'
            )

            # Filter out mitotic cells
            if remove_mitotics:
                # cell_stage = M0 indicates interphase cells
                df_meta = df_meta.loc[df_meta.cell_stage=='M0']
                df_features = df_features[df_features.index.isin(df_meta.index)]
                log.info(f"Shape of metadata without mitotics: {df_meta.shape}")

        # Outlier detection

        # Path to outliers manifest
        manifest_outliers_path = outliers_dir / 'manifest_outliers.csv'

        # Merged dataframe to compute outliers on
        df = pd.concat([df_meta,df_features], axis=1)
        
        # Compute outliers
        if not overwrite and manifest_outliers_path.is_file():
            log.info("Using pre-detected outliers.")
            df_outliers = pd.read_csv(manifest_outliers_path, index_col='CellId')
        else:
            log.info("Computing outliers...")
            df_outliers = outliers_removal(
                df = df,
                output_dir = outliers_dir,
                log = log,
                detect_based_on_structure_features = not no_structural_outliers
            )
            df_outliers.to_csv(manifest_outliers_path)

        # Generate outliers table
        df.loc[df_outliers.index, 'Outlier'] = df_outliers['Outlier']

        # Save a data table with detected outliers
        dataset_summary_table(
            df = df,
            levels = ['Outlier'],
            factor = 'structure_name',
            save = tables_dir / 'outliers'
        )
        df = df.loc[df.Outlier == 'No']
        df = df.drop(columns=['Outlier'])

        log.info(f"Shape of data without outliers: {df.shape}")
        
        # Save a data table stratifyed by metadata if information
        # is available
        metadata_columns = [
                'WorkflowId',
                'meta_imaging_mode',
                'meta_fov_position'
        ]
        if pd.Series(metadata_columns).isin(df.columns).all():
            dataset_summary_table(
                df = df,
                levels = metadata_columns,
                factor = 'structure_name',
                save = tables_dir / 'main',
            )

        # Dimensionality reduction and shape space calculation

        # Perform principal component analysis
        dir_pca = self.step_local_staging_dir / 'pca'
        dir_pca.mkdir(parents=True, exist_ok=True)
        dir_avgshape = self.step_local_staging_dir / 'avgshape'
        dir_avgshape.mkdir(parents=True, exist_ok=True)
        # Three types of shape space:
        # DNA: generated by nuclear coeffcients only
        # MEM: generated by cell coeffcients only
        # MEM: generated by cell and nuclear coefficients
        # IMportantly these shape spaces share the same
        # alignment procedure defined in the compute_features
        # step (lines 81-98).
        features_to_use = {
            # Uncomment for shape space with nuclear SHE coefficients only
            #'DNA': ['dna_shcoeffs_L'],
            # Uncomment for shape space with cell SHE coefficients only
            #'MEM': ['mem_shcoeffs_L'],
            # Shape space with cell and nuclear SHE coefficients
            'DNA_MEM': ['dna_shcoeffs_L', 'mem_shcoeffs_L']
        }
        
        df_shapemode_paths = pd.DataFrame([])
        for prefix, feature_prefixes in features_to_use.items():

            log.info(f"[{prefix}] - PCA on {', '.join(feature_prefixes)}")

            features = [
                fn for fn in df.columns if any(word in fn for word in feature_prefixes)
            ]

            # PCA
            df, pc_names, pca = pca_analysis(
                df = df,
                feature_names = features,
                prefix = prefix,
                npcs_to_calc = 8,
                save = dir_pca / f'pca_{prefix}'
            )

            # Make plot of PC cross correlations
            paired_correlation(
                df = df,
                features = pc_names,
                save = dir_pca / f'correlations_{prefix}'
            )

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
                    nbins = 9,
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
                    plot_limits = [-150, 150, -80, 80],
                )

                df_shapemode_paths = df_shapemode_paths.append(df_paths, ignore_index=True)

                import pdb; pdb.set_trace()

        # Save datafrme with path for meshes and gifs
        dataframe_path = self.step_local_staging_dir / 'shapemode.csv'
        df_shapemode_paths.to_csv(dataframe_path)
        # Save manifest
        self.manifest = df
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
