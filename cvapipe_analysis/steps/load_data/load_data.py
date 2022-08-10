#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

from cvapipe_analysis.tools import general, controller
from .load_data_tools import DataLoader

log = logging.getLogger(__name__)

class LoadData(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        staging,
        ignore_raw_data = False,
        **kwargs
        ):

        config = general.load_config_file()
        config["project"]["local_staging"] = staging
        control = controller.Controller(config)

        loader = DataLoader(control)
        if ignore_raw_data:
            loader.disable_download_of_raw_data()
        df = loader.load(kwargs)

        df.to_csv(Path(staging) / f"{self.step_name}/manifest.csv")
