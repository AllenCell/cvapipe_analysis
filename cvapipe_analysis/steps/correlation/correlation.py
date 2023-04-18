#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

from tqdm import tqdm

from ...tools import io, general, cluster, shapespace
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
        staging: Union[str, Path],
        verbose: Optional[bool]=False,
        distribute: Optional[bool]=False,
        **kwargs):

        step_dir = Path(staging) / self.step_name

        with general.configuration(step_dir) as control:

            control.create_step_subdirs(step_dir, ["values"])

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            space = shapespace.ShapeSpace(control)
            if control.get_number_of_map_points() == 1:
                # Do not remove extreme points when working on
                # matched datasets for which the number of bins
                # equals 1 (consistency with previous analysis).
                space.set_remove_extreme_points(False)
            space.execute(df)
            variables = control.get_variables_values_for_aggregation()
            df_agg = space.get_aggregated_df(variables, include_cellIds=True)
            df_sphere = space.get_cells_inside_ndsphere_of_radius()
            df_agg = df_agg.append(df_sphere, ignore_index=True)

            agg_cols = [f for f in df_agg.columns if f not in ["CellIds", "structure"]]
            df_agg = df_agg.groupby(agg_cols).agg({"CellIds": sum})
            df_agg = df_agg.reset_index()

            if distribute:

                distributor = cluster.CorrelationDistributor(self, control)
                distributor.set_data(df_agg)
                distributor.distribute_by_row()
                distributor.jobs_warning()
                return None

            calculator = CorrelationCalculator(control)
            for _, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                '''Concurrent processes inside. Do not use concurrent here.'''
                calculator.execute(row)

