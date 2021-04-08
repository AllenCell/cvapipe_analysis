#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

from cvapipe_analysis.tools import io, general
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
    def run(self, debug: bool=False, **kwargs):

        with general.configuration(self.step_local_staging_dir) as control:
            
            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            log.info(f"Manifest: {df.shape}")

            for folder in ["pca", "avgshape"]:
                save_dir = self.step_local_staging_dir/folder
                save_dir.mkdir(parents=True, exist_ok=True)

            calculator = ShapeModeCalculator(control)
            calculator.set_data(df)
            calculator.execute()

        return
