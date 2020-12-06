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

    @log_run_params
    def run(self, debug=False, **kwargs):

        np.random.seed(666)

        """
        ---------------------------
        DOWNLOAD DATASET FROM QUILT
        ---------------------------
        """

        # Download raw CSV with feature from Quilt in case it is not found locally
        # The feature calculation will be incorporated as an extra step in this
        # workflow
        path_to_manifest = f"{self.step_local_staging_dir}/allcells_shapespace1.csv"
        if not os.path.exists(path_to_manifest):

            _ = quilt3.Package.browse(
                name="matheus/cell_shape_variation_shapespace1",
                registry="s3://allencell-internal-quilt",
            ).fetch(self.step_local_staging_dir)

        """
        --------------
        PRE-PROCESSING
        --------------
        """

        # Read the dataframe
        df = pd.read_csv(path_to_manifest, index_col=0, nrows=(1024 if debug else None))
        print(f"Shape of raw data: {df.shape}")

        # Save tava table with number of mitotic vs. intephase cells
        df["is_mitotic"] = df.cell_stage != "M0"
        dataset_summary_table(
            df=df,
            levels=["is_mitotic"],
            factor="structure_name",
            rank_factor_by="meta_fov_image_date",
            save=self.step_local_staging_dir / "allcells",
        )
        # Exclude mitotics
        df = df.loc[df.is_mitotic == False]
        df = df.drop(columns=["is_mitotic"])

        print(f"Shape of data without mitotis: {df.shape}")

        # Perform outlier detection based on Gaussian kernel estimation
        dir_output_outliers = self.step_local_staging_dir / "outliers"
        dir_output_outliers.mkdir(parents=True, exist_ok=True)
        if True:
            df_outliers = outliers_removal(df=df, output_dir=dir_output_outliers)
            df_outliers.to_csv(self.step_local_staging_dir / "manifest_outliers.csv")
        else:
            df_outliers = pd.read_csv(
                self.step_local_staging_dir / "manifest_outliers.csv", index_col=0
            )
            try:
                df_outliers = df_outliers.set_index("CellId", drop=True)
            except:
                pass

        df.loc[df_outliers.index, "Outlier"] = df_outliers["Outlier"]

        # Save a data table with detected outliers
        dataset_summary_table(
            df=df,
            levels=["Outlier"],
            factor="structure_name",
            rank_factor_by="meta_fov_image_date",
            save=self.step_local_staging_dir / "outliers",
        )
        df = df.loc[df.Outlier == "No"]
        df = df.drop(columns=["Outlier"])

        print(f"Shape of data without outliers: {df.shape}")

        # Save a data table stratifyed by metadata
        dataset_summary_table(
            df=df,
            levels=["WorkflowId", "meta_imaging_mode", "meta_fov_position"],
            factor="structure_name",
            rank_factor_by="meta_fov_image_date",
            save=self.step_local_staging_dir / "manifest",
        )

        # TEST ALTERNATIVE SHAPE SPACES
        # df = df.loc[df.edge_flag==1]
        # print(df.shape)
        # import pdb; pdb.set_trace()

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
