#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm
from .aggregation_tools import Aggregator
from ...tools import io, general, cluster, shapespace

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
        staging: Union[str, Path],
        verbose: Optional[bool]=False,
        distribute: Optional[bool]=False,
        **kwargs):

        step_dir = Path(staging) / self.step_name

        with general.configuration(step_dir) as control:

            control.create_step_subdirs(step_dir, ["repsagg", "aggmorph"])

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            space = shapespace.ShapeSpace(control)
            space.execute(df)
            variables = control.get_variables_values_for_aggregation()
            df_agg = space.get_aggregated_df(variables, include_cellIds=True)
            df_sphere = space.get_cells_inside_ndsphere_of_radius()
            df_agg = pd.concat([df_agg, df_sphere])
            # Slurm can only handle arrays with max size 10K. Slurm will throw an
            # error if df_agg is larger than that.
            df_agg = df_agg.reset_index(drop=True)

            if distribute:

                distributor = cluster.AggregationDistributor(self, control)
                distributor.set_data(df_agg)
                '''Setting chunk size to 1 here so that each job has to generate
                a single file. Otherwise Slurm crashes for reasons that I don't
                yet know. It seems to me that aggregation_tools.py is leaking
                memory. To be investigated...'''
                distributor.set_chunk_size(1)
                distributor.distribute()
                distributor.jobs_warning()

                return None

            aggregator = Aggregator(control)
            if verbose: 
                aggregator.set_verbose_mode_on()
            for _, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                '''Concurrent processes inside. Do not use concurrent here.'''
                aggregator.execute(row)
                
        return