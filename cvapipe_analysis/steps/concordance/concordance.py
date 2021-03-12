#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import pandas as pd

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Concordance(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        **kwargs
    ):

        # Load config file
        config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        
        # Load parameterization dataframe
        path_to_agg_manifest = self.project_local_staging_dir / 'aggregation/manifest.csv'
        if not path_to_agg_manifest.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_agg_manifest)
        df_agg = pd.read_csv(path_to_agg_manifest)
        log.info(f"Shape of agg manifest: {df_agg.shape}")
        
        '''
        # Save manifest
        self.manifest = df_hyperstacks_paths
        manifest_path = self.step_local_staging_dir / 'manifest.csv'
        self.manifest.to_csv(manifest_path)

        return manifest_path
        '''