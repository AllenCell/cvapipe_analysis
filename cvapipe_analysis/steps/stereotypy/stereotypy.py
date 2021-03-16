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
from .stereotypy_tools import StereotypyCalculator

import pdb;
tr = pdb.set_trace

log = logging.getLogger(__name__)

class Stereotypy(Step):
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
        **kwargs
    ):

        # Load configuration file
        config = general.load_config_file()
        
        # Make necessary folders
        for folder in ['values', 'plots']:
            save_dir = self.step_local_staging_dir / folder
            save_dir.mkdir(parents=True, exist_ok=True)

        # Create all combinations of parameters
        space = shapespace.ShapeSpace(config)
        space.load_shape_space_axes()
        df_agg = pd.DataFrame()
        for shapemode in space.iter_shapemodes(config):
            space.set_active_axis(shapemode, digitize=True)
            for b in space.iter_bins(config):
                space.set_active_bin(b)
                for struct in space.iter_structures(config):
                    space.set_active_structure(struct)
                    df_tmp = space.iter_param_values_as_dataframe(
                        config, [
                            ('aggtype', ['avg']),
                            ('intensity', space.iter_intensities),
                            ('structure_name', [struct]),
                            ('shapemode', [shapemode]),
                            ('bin', [b]),
                            ('CellIds', [space.get_active_cellids()])
                        ]
                    )
                    df_agg = df_agg.append(df_tmp, ignore_index=True)

        if distribute:
            
            nworkers = config['resources']['nworkers']            
            distributor = cluster.StereotypyDistributor(df_agg, nworkers)
            distributor.distribute(config, log)

            log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")            
            return None

        calculator = StereotypyCalculator(config)    
        for index, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
            '''Concurrent processes inside. Do not use concurrent here.'''
            df_agg.loc[index,'PathToStereotypyFile'] = calculator.execute(row)

        df_results = calculator.load_results_in_single_dataframe()
        
        for intensity in config['parameterization']['intensities']:
            pmaker = plotting.StereotypyPlotMaker(config)
            pmaker.set_dataframe(df_results)
            pmaker.filter_dataframe({
                'intensity': intensity,
                'shapemode': 'DNA_MEM_PC1',
                'bin': 5
            })
            pmaker.execute(save_as=f"MeanCell-{intensity}")
            pmaker.set_max_number_of_pairs(300)
            pmaker.execute(save_as=f"MeanCell-{intensity}-N300")
            
        self.manifest = df_agg
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
