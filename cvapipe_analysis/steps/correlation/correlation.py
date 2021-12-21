#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io as skio
from joblib import Parallel, delayed

from cvapipe_analysis.tools import io, general, cluster, shapespace
from .correlation_tools import CorrelationCalculator

log = logging.getLogger(__name__)

class Correlation(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        distribute: Optional[bool] = False,
        **kwargs
    ):

        with general.configuration(self.step_local_staging_dir) as control:

            for folder in ['values']:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            space = shapespace.ShapeSpace(control)
            space.execute(df)
            variables = control.get_variables_values_for_aggregation()
            df_agg = space.get_aggregated_df(variables, True)
            df_agg = space.sample_cell_ids(df_agg, 1000)
            agg_cols = [f for f in df_agg.columns if f not in ["CellIds", "structure"]]
            df_agg = df_agg.groupby(agg_cols).agg({"CellIds": sum})
            df_agg = df_agg.reset_index()

            if distribute:

                distributor = cluster.CorrelationDistributor(control)
                distributor.set_data(df_agg)
                distributor.distribute_by_row()
                log.info(
                    f"Multiple jobs have been launched. Please come back when the calculation is complete.")
                return None

            calculator = CorrelationCalculator(control)
            for _, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                '''Concurrent processes inside. Do not use concurrent here.'''
                calculator.execute(row)

