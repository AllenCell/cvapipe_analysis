#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import concurrent
import numpy as np
import pandas as pd

from tqdm import tqdm

from cvapipe_analysis.tools import general, cluster, shapespace, plotting
from .concordance_tools import ConcordanceCalculator

import pdb;
tr = pdb.set_trace

log = logging.getLogger(__name__)

class Concordance(Step):
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
        space = shapespace.ShapeSpaceBasic(config)
        df_agg = space.iter_param_values_as_dataframe(
            config, [
                ('aggtype', ['avg']),
                ('intensity', space.iter_intensities),
                ('structure_name1', space.iter_structures),
                ('structure_name2', space.iter_structures),
                ('shapemode', space.iter_shapemodes),
                ('bin', space.iter_bins)
            ], [('structure_name1', 'structure_name2')]
        )

        if distribute:
            
            nworkers = config['resources']['nworkers']            
            distributor = cluster.ConcordanceDistributor(df_agg, nworkers)
            distributor.distribute(config, log)

            log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")
            return None
                        
        calculator = ConcordanceCalculator(config)            
        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            paths = list(executor.map(calculator.execute, [row for _,row in df_agg.iterrows()]))
        df_agg['PathToConcordanceFile'] = paths
        
        log.info(f"Loading results...")
        
        df_results = calculator.load_results_in_single_dataframe()

        log.info(f"Generating plots...")
        
        space  = shapespace.ShapeSpaceBasic(config)
        pmaker = plotting.ConcordancePlotMaker(config)
        pmaker.set_dataframe(df_results)
        for intensity in tqdm(space.iter_intensities(config)):
            for shapemode in space.iter_shapemodes(config):
                pmaker.filter_dataframe({'intensity':intensity, 'shapemode':shapemode, 'bin':[5]})
                pmaker.execute(display=False)
                pmaker.filter_dataframe({'intensity':intensity, 'shapemode':shapemode, 'bin':[1,9]})
                pmaker.execute(display=False)
        
        self.manifest = df_agg
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)
        return manifest_path
