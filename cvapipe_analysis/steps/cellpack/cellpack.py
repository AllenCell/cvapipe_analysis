#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import warnings
from typing import Dict, List, Optional, Union

import pdb
import concurrent
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage import io as skio
from aicscytoparam import cytoparam
from datastep import Step, log_run_params
from aicsimageio import AICSImage, writers

from cvapipe_analysis.tools import general, cluster, shapespace
from .cellpack_tools import ObjectCollector

tr = pdb.set_trace
log = logging.getLogger(__name__)


class Cellpack(Step):
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

        config = general.load_config_file()

        save_dir = self.step_local_staging_dir / "data"
        save_dir.mkdir(parents=True, exist_ok=True)

        space = shapespace.ShapeSpace(config)
        space.load_shape_space_axes()
        space.load_shapemode_manifest()

        collector = ObjectCollector(config)

        structure = "SLC25A17"

        for shapemode in ['DNA_MEM_PC1']:
            for b in [2, 8]:
                prefix = f"{shapemode}_B{b}"
                space.set_active_axis(shapemode, digitize=True)
                space.set_active_structure(structure)
                space.set_active_bin(b)
                dna = space.get_dna_mesh_of_bin(b)
                mem = space.get_mem_mesh_of_bin(b)
                seg = cytoparam.voxelize_meshes([mem, dna])
                save_as = save_dir / f"{prefix}.tif"
                with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
                    writer.save(seg[0], dimension_order='ZYX', image_name=save_as.stem)

                CellIds = space.get_active_cellids()
                with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
                    objs = list(tqdm(executor.map(
                        collector.collect_segmented_objects,
                        [space.meta.loc[CellId] for CellId in CellIds]), total=len(CellIds)
                    ))

                objs = [o for obj in objs for o in obj]
                img = collector.pack_objs(objs)
                collector.save_img(img, f"{structure}_{prefix}_ncells_{len(CellIds)}")

        return None
