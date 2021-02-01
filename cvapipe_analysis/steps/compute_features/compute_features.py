#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict

import pandas as pd
from tqdm import tqdm
from datastep import Step, log_run_params
from .compute_features_tools import get_segmentations, get_features

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class ComputeFeatures(Step):

    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        debug=False,
        **kwargs):

        # Load manifest from previous step
        df = pd.read_csv(
            self.project_local_staging_dir/'loaddata/manifest.csv',
            index_col = 'CellId'
        )
        
        # Keep only the columns that will be used from now on
        columns_to_keep = ['crop_raw', 'crop_seg', 'name_dict']
        df = df[columns_to_keep]
        
        # Sample the dataset if running debug mode
        if debug:
            df = df.sample(n=8, random_state=666)

        df_features = pd.DataFrame([])
        for index in tqdm(df.index):
            
            # Find the correct segmentation for nucleus,
            # cell and structure
            channels = df.at[index,'name_dict']
            seg_dna, seg_mem, seg_str = get_segmentations(
                folder = self.project_local_staging_dir/'loaddata',
                path_to_seg = df.at[index,'crop_seg'],
                channels = eval(channels)['crop_seg']
            )
            
            # Compute nuclear features
            features_dna = get_features (
                input_image = seg_dna,
                input_reference_image = seg_mem
            )

            # Compute cell features
            features_mem = get_features (
                input_image = seg_mem,
                input_reference_image = seg_mem
            )

            # Compute structure features
            features_str = get_features (
                input_image = seg_str,
                input_reference_image = None,
                compute_shcoeffs = False
            )

            # Append prefix to features names
            features_dna = dict(
                (f'dna_{key}',value) for (key,value) in features_dna.items()
            )
            features_mem = dict(
                (f'mem_{key}',value) for (key,value) in features_mem.items()
            )
            features_str = dict(
                (f'str_{key}',value) for (key,value) in features_str.items()
            )
            
            # Concatenate all features for this cell
            features = features_dna.copy()
            features.update(features_mem)
            features.update(features_str)
            features = pd.Series(features, name=index)
            
            # Add to features dataframe
            df_features = df_features.append(features)
            
        df_features.index = df_features.index.rename('CellId')
            
        self.manifest = df_features
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)
            
        return manifest_save_path
