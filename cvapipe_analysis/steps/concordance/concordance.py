#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

import concurrent
from tqdm import tqdm
from cvapipe_analysis.tools import io, general, cluster, shapespace, plotting
from .concordance_tools import ConcordanceCalculator

log = logging.getLogger(__name__)


class Concordance(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(self, distribute: Optional[bool] = False, **kwargs):

        with general.configuration(self.step_local_staging_dir) as control:

            for folder in ['values', 'plots']:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            space = shapespace.ShapeSpace(control)
            space.execute(df)
            variables = control.get_variables_values_for_aggregation()
            variables = control.duplicate_variable(variables, "structure")
            df_agg = space.get_aggregated_df(variables, False)

            if distribute:

                distributor = cluster.ConcordanceDistributor(control)
                distributor.set_data(df_agg)
                distributor.distribute()
                log.info(
                    f"Multiple jobs have been launched. Please come back when the calculation is complete.")

                return None

            calculator = ConcordanceCalculator(control)
            with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
                executor.map(calculator.execute, [row for _, row in df_agg.iterrows()])

            log.info(f"Loading results...")

            df_results = calculator.load_results_in_single_dataframe()

            log.info(f"Generating plots...")

            pmaker = plotting.ConcordancePlotMaker(control)
            pmaker.set_dataframe(df_results)
            for alias in tqdm(control.get_aliases_to_parameterize()):
                for shape_mode in control.get_shape_modes():
                    mpId = control.get_center_map_point_index()
                    pmaker.filter_dataframe(
                        {'alias': alias, 'shape_mode': shape_mode, 'mpId': [mpId]})
                    pmaker.execute(display=False)
                    mpIds = control.get_extreme_opposite_map_point_indexes()
                    pmaker.filter_dataframe(
                        {'alias': alias, 'shape_mode': shape_mode, 'mpId': mpIds})
                    pmaker.execute(display=False)
