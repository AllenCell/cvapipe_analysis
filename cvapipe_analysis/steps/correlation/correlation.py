#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import errno
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datastep import Step, log_run_params

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io as skio
from joblib import Parallel, delayed

from cvapipe_analysis.tools import io, general, controller, bincorr

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
        gene,
        distribute: Optional[bool] = False,
        **kwargs
    ):

        def read_rep(eindex):
            i, index = eindex
            rep = device.read_parameterized_intensity(index)
            rep = rep.astype(bool).flatten()
            rep[0] = True
            reps[i] = rep
            return

        def get_next_pair():
            for i in range(ncells):
                for j in range(i+1, ncells):
                    yield (i, j)

        def correlate_ij(ij):
            i, j = ij
            corrs[i, j] = corrs[j, i] = bincorr.calculate(reps[i], reps[j], 532610)
            return

        def sample_by_factors(df, factors, n):
            df_sample = pd.DataFrame([])
            for factors, df_factor in df.groupby(factors):
                df_sample = df_sample.append(df_factor.sample(n, replace=False, random_state=39), ignore_index=False)
            return df_sample

        with general.configuration(self.step_local_staging_dir) as control:

            config = general.load_config_file("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis")
            control = controller.Controller(config)
            device = io.LocalStagingIO(control)

            df = device.load_step_manifest("preprocessing")

            df = df.loc[df.structure_name==gene]
            #df = sample_by_factors(df, ["structure_name"], 1000)
            print(f"Running gene {gene}. Dataframe of size: {df.shape}")
            print(df.groupby("structure_name").size())

            ncells = len(df)
            rep_length = 532610

            reps = np.zeros((ncells, rep_length), dtype=bool)
            repsize = int(sys.getsizeof(reps)) / float(1 << 20)
            print(f"Representations shape: {reps.shape} ({reps.dtype}, {repsize:.1f}Mb)")

            corrs = np.zeros((ncells, ncells), dtype=np.float32)
            corrssize = int(sys.getsizeof(corrs)) / float(1 << 20)
            print(f"Correlations shape: {corrs.shape} ({corrs.dtype}, {corrssize:.1f}Mb)")

            ncores = control.get_ncores()
            print(f"Loading representations using {ncores} cores...")

            _ = Parallel(n_jobs=ncores, backend="threading")(
                delayed(read_rep)(eindex)
                for eindex in tqdm(enumerate(df.index), total=ncells)
            )

            npairs = int(ncells*(ncells-1)/2)
            _ = Parallel(n_jobs=ncores, backend="threading")(
                delayed(correlate_ij)(ij)
                for ij in tqdm(get_next_pair(), total=npairs)
            )

            skio.imsave(f"{control.get_staging()}/correlation/matrix_{gene}.tif", corrs)
            df[["structure_name"]].to_csv(f"{control.get_staging()}/correlation/matrix_{gene}_index.csv")
