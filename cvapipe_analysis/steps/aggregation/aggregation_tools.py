import os
import vtk
import json
import psutil
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
from aicscytoparam import cytoparam
from aicsimageio import AICSImage, writers
from typing import Dict, List, Optional, Union
from aics_dask_utils import DistributedHandler
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import concurrent

from cvapipe_analysis.tools import general, cluster, shapespace

class Aggregator(general.DataProducer):
    """
    The goal of this class is to have a combination of
    parameters as input, including some CellIds. The
    corresponding cells have their parameterized intensity
    representation morphed into the appropriated shape
    space shape according to the input parameters.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    
    subfolder = 'aggregation/aggmorph'
    
    def __init__(self, config):
        super().__init__(config)
    
    def set_shape_space(self, space):
        self.space = space
        self.load_parameterization_manifest()
    
    def aggregate_parameterized_intensities(self):
        with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
            pints = list(
                executor.map(self.read_parameterized_intensity, self.CellIds))
        agg_pint = self.agg_func(np.array(pints), axis=0)
        channel_id = self.get_available_parameterized_intensities().index(self.row.intensity)
        self.aggregated_parameterized_intensity = agg_pint[channel_id].copy()
    
    def set_agg_function(self):
        if self.row.aggtype == 'avg':
            self.agg_func = np.mean
        elif self.row.aggtype == 'std':
            self.agg_func = np.std
        else:
            raise ValueError(f"Aggregation type {self.row.aggtype} is not implemented.")
    
    def aggregate(self, row):
        self.digest_row_with_cellids(row)
        self.set_agg_function()
        self.aggregate_parameterized_intensities()
        self.space.set_active_axis(row.shapemode)
        
    def voxelize_and_parameterize_shapemode_shape(self):       
        mesh_dna = self.space.get_dna_mesh_of_bin(self.row.bin)
        mesh_mem = self.space.get_mem_mesh_of_bin(self.row.bin)
        domain, origin = cytoparam.voxelize_meshes([mesh_mem, mesh_dna])
        coords_param, _ = cytoparam.parameterize_image_coordinates(
            seg_mem = (domain>0).astype(np.uint8),
            seg_nuc = (domain>1).astype(np.uint8),
            lmax = 16,
            nisos = [32,32]
        )
        self.domain = domain
        self.origin = origin
        self.coords_param = coords_param
    
        return
        
    def morph_on_shapemode_shape(self):
        self.voxelize_and_parameterize_shapemode_shape()
        self.morphed = cytoparam.morph_representation_on_shape(
            img=self.domain,
            param_img_coords=self.coords_param,
            representation=self.aggregated_parameterized_intensity
        )
        self.morphed = np.stack([self.domain, self.morphed])
    
        return
    
    @staticmethod
    def get_output_file_name(row):
        return f"{row.aggtype}-{row.intensity}-{row.structure_name}-{row.shapemode}-B{row.bin}.tif"
    
    def get_rel_aggrep_file_path_as_str(self, row):
        file_name = self.get_aggrep_file_name(row)
        return f"{self.abs_path_local_staging.name}/aggregation/repsagg/{file_name}"
    
    def save(self):
        n = len(self.CellIds)
        save_as = self.get_rel_output_file_path_as_str(self.row)
        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(
                self.morphed,
                dimension_order='CZYX',
                image_name=f"N{n}",
                channel_names = ['domain', Path(save_as).stem]
            )
        aggrep = self.aggregated_parameterized_intensity
        save_as = self.get_rel_aggrep_file_path_as_str(self.row)
        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(
                aggrep.reshape(1,*aggrep.shape),
                dimension_order='ZYX',
                image_name=f"N{n}"
            )
        
        return save_as

    def workflow(self, row):
        rel_path_to_output_file = self.check_output_exist(row)
        if (rel_path_to_output_file is None) or self.config['project']['overwrite']:
            self.set_row(row)
            try:
                self.aggregate(row)
                self.morph_on_shapemode_shape()
                rel_path_to_output_file = self.save()
            except:
                rel_path_to_output_file = None
        self.status(row.name, rel_path_to_output_file)
        return rel_path_to_output_file

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Batch aggregation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())
    
    df = pd.read_csv(args['csv'], index_col=0)

    config = general.load_config_file()
    
    space = shapespace.ShapeSpaceBasic(config)
    aggregator = Aggregator(config)
    aggregator.set_shape_space(space)
    for _, row in df.iterrows():
        '''Concurrent processes inside. Do not use concurrent here.'''
        aggregator.workflow(row)

            
'''
class AggHyperstack:
    
    def __init__(self, pc_name, agg, intensity):
        self. pc_name = pc_name
        self.agg = agg
        self.intensity = intensity
        
    def set_path_to_agg_and_shapemode_folders(self, agg_path, smode_path):
        self.agg_folder = agg_path
        self.smode_folder = smode_path
        
        self.space = shapespace.ShapeSpaceBasic()
        self.space.link_results_folder(smode_path)
        self.space.set_active_axis(self.pc_name)

    def get_path_to_agg_file(self, b, struct):
        path = self.agg_folder / f"{self.agg}-{self.intensity}-{struct}-{self.pc_name}-B{b}.tif"
        return path
        
    def create(self, save_as, config):
        nbins = config['pca']['number_map_points']
        nstrs = len(config['structures']['genes'])
        hyperstack = []
        for b in tqdm(range(1,1+nbins)):
            imgs = []
            for struct in config['structures']['genes']:
                path = self.get_path_to_agg_file(b, struct)
                if path.is_file():
                    img = AICSImage(path).data.squeeze()
                else:
                    img = None
                imgs.append(img)

            mesh_dna = self.space.get_dna_mesh_of_bin(b)
            mesh_mem = self.space.get_mem_mesh_of_bin(b)
            domain, _ = cytoparam.voxelize_meshes([mesh_mem, mesh_dna])

            stack = np.zeros((nstrs,*domain.shape), dtype=np.float32)
            for sid, struct in enumerate(config['structures']['genes']):
                if imgs[sid] is not None:
                    stack[sid] = imgs[sid].copy()
            stack = np.vstack([stack, domain.reshape(1, *domain.shape)])
            hyperstack.append(stack)
            
        # Calculate largest czyx bounding box across bins
        shapes = np.array([stack.shape for stack in hyperstack])
        lbb = shapes.max(axis=0)

        # Pad all stacks so they end up with similar shapes
        for b in range(nbins):
            stack_shape = np.array(hyperstack[b].shape)
            pinf = (0.5 * (lbb - stack_shape)).astype(np.int)
            psup = lbb - (stack_shape + pinf)
            pad = [(i,s) for i,s in zip(pinf, psup)]
            hyperstack[b] = np.pad(hyperstack[b], list(pad))
        # Final hyperstack. This variable has ~4Gb for the
        # full hiPS single-cell images dataset.
        hyperstack = np.array(hyperstack)

        # Save hyperstack
        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(
                hyperstack,
                dimension_order = 'TCZYX',
                image_name = f"{self.pc_name}_{self.intensity}_{self.agg}",
                channel_names = config['structures']['genes'] + ['domain']
            )

        return
'''