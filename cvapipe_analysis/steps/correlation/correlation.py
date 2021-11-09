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

            for folder in ['values', 'plots']:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            df = df.sample(100)
            
            if distribute:

                distributor = cluster.CorrelationDistributor(control)
                distributor.set_data(df)
                distributor.distribute_by_blocks()
                log.info(
                    f"Multiple jobs have been launched. Please come back when the calculation is complete.")

                return None

            calculator = CorrelationCalculator(control)
            calculator.set_indexes(df.index, df.index)
            # Does not call execute because does not run on row-basis
            calculator.workflow()

