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

from cvapipe_analysis.tools import io, general, cluster, shapespace, plotting
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

        with general.configuration(self.step_local_staging_dir) as control:

            for folder in ['values', 'plots']:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            space = shapespace.ShapeSpace(control)
            space.execute(df)
            variables = control.get_variables_values_for_aggregation()
            df_agg = space.get_aggregated_df(variables, True)

            if distribute:

                nworkers = control['resources']['nworkers']            
                distributor = cluster.StereotypyDistributor(df_agg, nworkers)
                distributor.distribute(control, log)

                log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")            
                return None

            calculator = StereotypyCalculator(control)
            for index, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                '''Concurrent processes inside. Do not use concurrent here.'''
                calculator.execute(row)

            log.info(f"Loading results...")

            df_results = calculator.load_results_in_single_dataframe()

            log.info(f"Generating plots...")

            pmaker = plotting.StereotypyPlotMaker(control)
            pmaker.set_dataframe(df_results)
            for alias in tqdm(control.get_aliases_to_parameterize()):
                for shape_mode in control.get_shape_modes():
                    mpId = control.get_center_map_point_index()
                    pmaker.filter_dataframe({'alias': alias, 'shape_mode': shape_mode, 'mpId': [mpId]})
                    pmaker.execute(display=False)


