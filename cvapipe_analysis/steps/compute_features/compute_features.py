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
    def run(self, distribute: Optional[bool]=False, **kwargs):

        with general.configuration(self.step_local_staging_dir) as control:

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("loaddata")
            log.info(f"Manifest: {df.shape}")

            save_dir = self.step_local_staging_dir/"cell_features"
            save_dir.mkdir(parents=True, exist_ok=True)

            if distribute:

                distributor = cluster.FeaturesDistributor(self, control)
                distributor.set_data(df)
                distributor.distribute()
                log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")

                return None

            calculator = FeatureCalculator(control)
            with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
                executor.map(calculator.execute, [row for _,row in df.iterrows()])

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
            self.manifest = df.merge(df_results, left_index=True, right_index=True, how="outer")
            manifest_path = self.step_local_staging_dir / 'manifest.csv'
            self.manifest.to_csv(manifest_path)

        return
