#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

from ...tools import io, general
from .shapemode_tools import ShapeModeCalculator

log = logging.getLogger(__name__)

class Shapemode(Step):

    def __init__(
        self,
        direct_upstream_tasks: List['Step'] = [],
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
            
            # control.create_step_subdirs(step_dir, ["pca", "avgshape"])

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            log.info(f"Manifest: {df.shape}")

            calculator = ShapeModeCalculator(control)
            if verbose: 
                calculator.set_verbose_mode_on()
            calculator.set_data(df)
            calculator.execute()

        return
