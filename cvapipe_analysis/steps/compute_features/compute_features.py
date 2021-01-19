#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Union, List, Dict
from aicsfeature.extractor import cell, cell_nuc, dna
from scipy.ndimage import gaussian_filter as ndf

import aicsimageio
import dask.dataframe as dd
import pandas as pd
from aics_dask_utils import DistributedHandler
from aicsimageio import AICSImage
from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SingleCellFeaturesResult(NamedTuple):
    cell_id: Union[int, str]
    path: Path


class SingleCellFeaturesError(NamedTuple):
    cell_id: int
    error: str


class DatasetFields:
    CellId = "CellId"
    CellIndex = "CellIndex"
    FOVId = "FOVId"
    CellFeaturesPath = "CellFeaturesPath"


###############################################################################

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
        debug=False,
        **kwargs):

        np.random.seed(666)
        
        return None
