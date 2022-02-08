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
            df_agg = space.get_aggregated_df(variables, include_cellIds=False)

            if distribute:

                distributor = cluster.ConcordanceDistributor(control)
                distributor.set_data(df_agg)
                distributor.distribute()
                log.info(
                    f"Multiple jobs have been launched. Please come back when the calculation is complete.")

                return None

            calculator = ConcordanceCalculator(control)
            calculator.set_row(df_agg.loc[df_agg.index[0]])
            calculator.workflow()
            with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
                executor.map(calculator.execute, [row for _, row in df_agg.iterrows()])

            log.info(f"Generating plots...")

            variables = control.get_variables_values_for_aggregation()
            df_agg = space.get_aggregated_df(variables, include_cellIds=False)
            df_agg =  df_agg.drop(columns=["structure"]).drop_duplicates().reset_index()

            for index, row in tqdm(df_agg.iterrows(), total=len(df_agg)):
                for mode in [True, False]:
                    pmaker = plotting.ConcordancePlotMaker(control)
                    pmaker.use_average_representations(mode)
                    pmaker.set_dataframe(df)
                    pmaker.set_row(row)
                    pmaker.execute(display=False)

