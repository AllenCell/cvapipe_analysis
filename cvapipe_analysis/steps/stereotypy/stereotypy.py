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
        distribute: Optional[bool] = False,
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
            df_agg = space.get_aggregated_df(variables, include_cellIds=False)
            '''Do not aggregate cells inside nD sphere when working with
            datasets with a single map point, which is the case of matched datasets'''
            if len(control.get_map_points()) > 1:
                variables.update({"shape_mode": ["NdSphere"], "mpId": [control.get_center_map_point_index()]})
                df_sphere = space.get_aggregated_df(variables, include_cellIds=False)
                df_agg = df_agg.append(df_sphere, ignore_index=True)
            df_agg =  df_agg.drop(columns=["structure"]).drop_duplicates().reset_index(drop=True)

            log.info(f"Generating plots...")

            for index, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                pmaker = plotting.StereotypyPlotMaker(control)
                pmaker.set_heatmap_min_max_values(-0.2, 0.2)
                pmaker.set_dataframe(df)
                pmaker.set_row(row)
                if row.mpId == 1:
                    pmaker.set_extra_values({"mpId": df_agg.mpId.unique().tolist()})
                pmaker.execute(display=False)
