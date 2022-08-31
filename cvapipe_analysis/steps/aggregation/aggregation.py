#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

from tqdm import tqdm
from cvapipe_analysis.tools import io, general, cluster, shapespace
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
    def run(self, distribute: Optional[bool]=False, **kwargs):

        with general.configuration(self.step_local_staging_dir) as control:

            for folder in ["repsagg", "aggmorph"]:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            space = shapespace.ShapeSpace(control)
            space.execute(df)
            variables = control.get_variables_values_for_aggregation()
            df_agg = space.get_aggregated_df(variables, include_cellIds=True)
            df_sphere = space.get_cells_inside_ndsphere_of_radius()
            df_agg = df_agg.append(df_sphere, ignore_index=True)

            if distribute:

                distributor = cluster.AggregationDistributor(self, control)
                distributor.set_data(df_agg)
                '''Setting chunk size to 1 here so that each job has to generate
                a single file. Otherwise Slurm crashes for reasons that I don't
                yet know. It seems to me that aggregation_tools.py is leaking
                memory. To be investigated...'''
                distributor.set_chunk_size(1)
                distributor.distribute()
                log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")

                return None

            aggregator = Aggregator(control)
            for index, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                '''Concurrent processes inside. Do not use concurrent here.'''
                aggregator.execute(row)
                
        return