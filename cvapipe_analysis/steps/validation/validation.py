#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

import concurrent
from .validation_tools import Validator
from ...tools import io, general, cluster, plotting

log = logging.getLogger(__name__)

class Validation(Step):
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

            control.create_step_subdirs(step_dir, ["output"])

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            log.info(f"Manifest: {df.shape}")

            df = df.sample(n=300, random_state=42)

            if distribute:
                # TBD
                return None

            validator = Validator(control)
            if verbose:
                validator.set_verbose_mode_on()
            with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
                executor.map(validator.execute, [row for _,row in df.iterrows()])

            log.info(f"Loading results...")
            df_error = validator.load_results_in_single_dataframe()
            
            pmaker = plotting.ValidationPlotMaker(control)
            pmaker.set_dataframe(df_error)
            pmaker.execute(display=False)

            df_error.to_csv(step_dir/"rec_error.csv", index=False)

        return
