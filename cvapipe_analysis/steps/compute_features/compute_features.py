#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

import concurrent
from cvapipe_analysis.tools import io, general, cluster
from .compute_features_tools import FeatureCalculator

log = logging.getLogger(__name__)

class ComputeFeatures(Step):
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

        with general.configuration(staging) as control:

            step_folder = control.create_step_dirs(self.step_name, ["cell_features"])

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("loaddata")
            log.info(f"Manifest: {df.shape}")

            if distribute:

                distributor = cluster.FeaturesDistributor(self, control)
                distributor.set_data(df)
                distributor.distribute()
                distributor.jobs_warning()

                return None

            calculator = FeatureCalculator(control)
            if verbose:
                calculator.set_verbose_mode_on()
            for _, row in df.iterrows():
                calculator.execute(row)

            log.info(f"Loading results...")
            df_results = calculator.load_results_in_single_dataframe()
            df_results = df_results.set_index('CellId')

            if len(df) != len(df_results):
                df_miss = df.loc[~df.index.isin(df_results.index)]
                log.info("Missing feature for indices:")
                for index in df_miss.index:
                    log.info(f"\t{index}")
                log.info(f"Total of {len(df_miss)} indices.")

            log.info(f"Saving manifest...")
            df = df.merge(df_results, left_index=True, right_index=True, how="outer")
            df.to_csv(step_folder/"manifest.csv")

        return
