#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import os
import quilt3
import numpy as np
import pandas as pd

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

        
        # Load feature dataframe
        path_to_metadata_manifest = self.project_local_staging_dir / 'loaddata/manifest.csv'
        df_meta = pd.read_csv(path_to_metadata_manifest, index_col='CellId')
        # Drop unwanted columns
        df_meta = df_meta[
            [c for c in df_meta.columns if any(s not in c for s in ['mem_','dna_','str_'])]
        ]
        log.info(f"Shape of metadata: {df_meta.shape}")

        # Load feature dataframe
        path_to_features_manifest = self.project_local_staging_dir / 'computefeatures/manifest.csv'
        df_features = pd.read_csv(path_to_features_manifest, index_col='CellId')
        log.info(f"Shape of features data: {df_features.shape}")
        
        # Make necessary folders
        tables_dir = self.step_local_staging_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # Perform outlier detection based on Gaussian kernel estimation
        outliers_dir = self.step_local_staging_dir / "outliers"
        outliers_dir.mkdir(exist_ok=True)

        # Mitotic removal
        
        if 'cell_stage' in df_meta.columns:
        
            # Generate table with number of mitotic vs. intephase cells
            dataset_summary_table(
                df = df_meta,
                levels = ["cell_stage"],
                factor = "structure_name",
                save = tables_dir / "cell_stage"
            )

            # Filter out mitotic cells
            if remove_mitotics:
                # cell_stage = M0 indicates interphase cells
                df_meta = df_meta.loc[df_meta.cell_stage=='M0']
                df_features = df_features[df_features.index.isin(df_meta.index)]
                log.info(f"Shape of metadata without mitotics: {df_meta.shape}")

        # Outlier detection

        # Path to outliers manifest
        manifest_outliers_path = outliers_dir / "manifest_outliers.csv"

        # Merged dataframe to compute outliers on
        df = pd.concat([df_meta,df_features], axis=1)

        # Compute outliers
        if not overwrite and manifest_outliers_path.is_file():
            log.info(f"Using pre-detected outliers.")
            df_outliers = pd.read_csv(manifest_outliers_path, index_col='CellId')
        else:
            log.info(f"Computing outliers...")
            df_outliers = outliers_removal(
                df = df,
                output_dir = outliers_dir,
                log = log,
                detect_based_on_structure_features = not no_structural_outliers
            )
            df_outliers.to_csv(manifest_outliers_path)

        # Generate outliers table
        df.loc[df_outliers.index, "Outlier"] = df_outliers["Outlier"]

        # Save a data table with detected outliers
        dataset_summary_table(
            df = df,
            levels = ["Outlier"],
            factor = "structure_name",
            save = tables_dir / "outliers"
        )
        df = df.loc[df.Outlier == "No"]
        df = df.drop(columns=["Outlier"])

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
                factor = "structure_name",
                save = tables_dir / "main",
            )

        """
        ------------------------
        DIMENSIONALITY REDUCTION
        ------------------------
        """

        # Perform principal component analysis
        dir_output_pca = self.step_local_staging_dir / "pca"
        dir_output_pca.mkdir(parents=True, exist_ok=True)
        dir_output_avgshape = self.step_local_staging_dir / "avgshape"
        dir_output_avgshape.mkdir(parents=True, exist_ok=True)
        features_to_use = {
            "DNA": ["dna_shcoeffs_L"],
            "MEM": ["mem_shcoeffs_L"],
            "DNA_MEM": ["dna_shcoeffs_L", "mem_shcoeffs_L"],
        }

        for prefix, feature_prefixes in features_to_use.items():

            print(f"[{prefix}] - PCA on {', '.join(feature_prefixes)}")

            features = [
                fn for fn in df.columns if any(word in fn for word in feature_prefixes)
            ]

            # PCA
            df, pc_names, pca = pca_analysis(
                df=df,
                feature_names=features,
                prefix=prefix,
                npcs_to_calc=8,
                npcs_to_show=8,
                save=f"{dir_output_pca}/pca_{prefix}",
            )

            # Make plot of PC cross correlations
            paired_correlation(
                df=df, features=pc_names, save=f"{dir_output_pca}/correlations_{prefix}"
            )

            """
            -----------------------
            SHAPE MODES CALCULATION
            -----------------------
            """

            # Calculates the average shapes along each PC
            for mode, pc_name in enumerate(pc_names):

                print(
                    f"\tCalculating average cell and nuclear shape for mode: {pc_name}"
                )

                # Create map points, bins e get all cells in each bin
                df_filtered, bin_indexes, (bin_centers, pc_std) = digitize_shape_mode(
                    df=df,
                    feature=pc_name,
                    nbins=9,
                    save=f"{dir_output_avgshape}/{pc_name}",
                )

                # Convert map points back to SH coefficients
                df_mappoints_shcoeffs = get_shcoeffs_from_pc_coords(
                    coords=bin_centers * pc_std,
                    pc=mode,
                    pca=pca,
                    coeff_names=features,
                )

                if "DNA" not in pc_name:
                    # Create fake DNA coeffs
                    df_tmp = df_mappoints_shcoeffs.copy()
                    df_tmp.columns = [f.replace("mem", "dna") for f in df_tmp.columns]
                    df_mappoints_shcoeffs = pd.concat(
                        [df_mappoints_shcoeffs, df_tmp], axis=1
                    )

                if "MEM" not in pc_name:
                    # Create fake MEM coeffs
                    df_tmp = df_mappoints_shcoeffs.copy()
                    df_tmp.columns = [f.replace("dna", "mem") for f in df_tmp.columns]
                    df_mappoints_shcoeffs = pd.concat(
                        [df_mappoints_shcoeffs, df_tmp], axis=1
                    )

                # Reconstruct cell and nuclear shape. Correct nucleus location and
                # save the meshes as VTK files
                animate_shape_modes_and_save_meshes(
                    df=df_filtered,
                    df_agg=df_mappoints_shcoeffs,
                    bin_indexes=bin_indexes,
                    feature=pc_name,
                    save=dir_output_avgshape,
                    fix_nuclear_position=False if prefix != "DNA_MEM" else True,
                    plot_limits=[-150, 150, -80, 80],
                )

        # Save manifest
        self.manifest = df
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv")
        print("Manifest saved.")

        return self.step_local_staging_dir / "manifest.csv"
