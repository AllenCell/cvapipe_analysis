#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm

from cvapipe_analysis.tools import general, cluster, shapespace
from .shapemode_tools import ShapeModeCalculator

from .outliers import outliers_removal
from .dim_reduction import pca_analysis
from .avgshape import digitize_shape_mode
from .avgshape import get_shcoeffs_from_pc_coords
from .avgshape import animate_shape_modes_and_save_meshes
from .plotting import paired_correlation, dataset_summary_table

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

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
        debug: bool = False,
        overwrite: bool = False,
        remove_mitotics: bool = True,
        no_structural_outliers: bool = False,
        **kwargs
    ):
        
        with general.configuration(self.step_local_staging_dir) as config:
            # Load parameterization dataframe
            path_to_loaddata_manifest = self.project_local_staging_dir / 'computefeatures/manifest.csv'
            df = pd.read_csv(path_to_loaddata_manifest, index_col='CellId')
            log.info(f"Manifest: {df.shape}")
            # Make necessary folders
            for folder in ['pca', 'avgshape']:
                save_dir = self.step_local_staging_dir / folder
                save_dir.mkdir(parents=True, exist_ok=True)

            calculator = ShapeModeCalculator(config)
            calculator.workflow(df)
        return
