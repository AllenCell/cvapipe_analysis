#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import numpy as np
import pandas as pd
from tqdm import tqdm

from cvapipe_analysis.tools import general, cluster, shapespace
from cvapipe_analysis.steps.shapemode.avgshape import digitize_shape_mode
from .aggregation_tools import Aggregator

log = logging.getLogger(__name__)

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
        distribute: Optional[bool]=False,
        overwrite: Optional[bool]=False,
        **kwargs
    ):

        # Load configuration file
        config = general.load_config_file()
        
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
        df = pd.read_csv(path_to_shapemode_manifest, index_col='CellId', low_memory=False)
        log.info(f"Shape of shape mode manifest: {df.shape}")

        # Merge the two dataframes (they do not have
        # necessarily the same size)
        df = df.merge(df_param[['PathToRepresentationFile']], left_index=True, right_index=True)

        # Also read the manifest with paths to VTK files
        path_to_shapemode = self.project_local_staging_dir / 'shapemode'
        
        # Make necessary folders
        agg_dir = self.step_local_staging_dir / 'aggregations'
        agg_dir.mkdir(parents=True, exist_ok=True)
        
        # Agg representations per cells and shape mode.
        # Here we use principal components gerated with cell
        # and nuclear SHE coefficients (DNA_MEM_PCx).
        PREFIX = config['aggregation']['aggregate_on']
        
        pc_names = [f for f in df.columns if PREFIX in f]
                
        if distribute:
            
            log.info(f"Saving dataframe for workers...")
            path_manifest = Path(".distribute/manifest.csv")
            df.to_csv(path_manifest)
            
            dist_agg = cluster.DistributeAggregation(df)
            dist_agg.set_rel_path_to_dataframe(path_manifest)
            dist_agg.set_rel_path_to_shapemode_results(path_to_shapemode)
            dist_agg.set_rel_path_to_output(agg_dir)
            dist_agg.distribute(config, log)

            log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")
            
            return None
        
        else:
                        
            space = shapespace.ShapeSpace(df[pc_names])            
            for pc_name in tqdm(pc_names):
                space.set_active_axis(pc_name)
                space.digitize_active_axis()
                space.link_results_folder(path_to_shapemode)
                for _, intensity in config['parameterization']['intensities']:
                    for agg in ['avg', 'std']:
                        for b, _ in space.iter_map_points():
                            indexes = space.get_indexes_in_bin(b)
                            for struct in tqdm(config['structures']['genes'], leave=False):
                                df_struct = df.loc[(df.index.isin(indexes))&(df.structure_name==struct)]
                                name = f"{agg}-{intensity}-{struct}-{pc_name}-B{b}.tif"
                                save_as = agg_dir / name
                                if overwrite or not save_as.is_file():
                                    if len(df_struct) > 0:
                                        Agg = Aggregator(pc_name, b, agg, intensity)
                                        Agg.set_config(config)
                                        Agg.set_shape_space(df_struct, space)
                                        Agg.morph_parameterized_intensity_on_shape(save_as)
        '''
        # Loop over shape modes
        df_hyperstacks_paths = pd.DataFrame([])
        for pc_idx, pc_name in enumerate(pc_names):

            log.info(f"Running PC: {pc_name}.")

            df_paths = create_5d_hyperstacks(
                df=df,
                df_paths=df_shapemode_paths,
                pc_names=pc_names,
                pc_idx=pc_idx,
                nbins=config['pca']['number_map_points'],
                save_dir=hyper_dir
            )
            
            df_hyperstacks_paths = df_hyperstacks_paths.append(df_paths, ignore_index=True)
            
        # Save manifest
        self.manifest = df_hyperstacks_paths
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
        '''
        
        return None