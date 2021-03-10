import os
import vtk
import json
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

from cvapipe_analysis.tools import general, shapespace
from cvapipe_analysis.steps.shapemode.avgshape import digitize_shape_mode

class AggregatorNew:
    
    def __init__(self, df, space):
        self.df = df.copy()
        self.space = space
        
    def read_parameterized_intensity(self, index, return_intensity_names=False):
        code = AICSImage(self.df.at[index,'PathToRepresentationFile'])
        intensity_names = code.get_channel_names()
        code = code.data.squeeze()
        if return_intensity_names:
            return code, intensity_names
        return code

    def set_output_folder(self, path):
        self.output_folder = path

    def get_structures_of_current_indexes(self):
        return self.df.loc[self.row.CellIds,"structure_name"].unique()
        
    def get_output_file_name(self):
        return f"{self.row.aggtype}-{self.row.intensity}-{self.struct}-{self.row.shapemode}-B{self.row.bin}.tif"
    
    def get_available_intensities(self):
        _, channel_names = self.read_parameterized_intensity(self.df.index[0], True)
        return channel_names
    
    def aggregate_parameterized_intensities(self):
        
        intensity_names = self.get_available_intensities()
        
        N_CORES = len(os.sched_getaffinity(0))
        with concurrent.futures.ProcessPoolExecutor(N_CORES) as executor:
            pints = list(
                executor.map(self.read_parameterized_intensity, self.row.CellIds))
        agg_pint = self.agg_func(np.array(pints), axis=0)
        channel_id = intensity_names.index(self.row.intensity)
        self.aggregated_parameterized_intensity = agg_pint[channel_id]
    
    def set_agg_function(self):
        if self.row.aggtype == 'avg':
            self.agg_func = np.mean
        elif self.row.aggtype == 'std':
            self.agg_func = np.std
        else:
            raise ValueError(f"Aggregation type {self.row.aggtype} is not implemented.")
    
    def aggregate(self, row):
        self.row = row
        struct = self.get_structures_of_current_indexes()
        if len(struct) > 1:
            raise ValueError(f"Multiple structures found in this aggregation: {struct}")
        self.struct = struct[0]
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

        return
    
    def save(self):
        n = len(self.row.CellIds)
        save_as = self.output_folder / self.get_output_file_name()
        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(
                self.morphed,
                dimension_order='ZYX',
                image_name=f"{save_as.stem}-N{n}"
            )
        
        return save_as

    
class Aggregator:

    config = None
    
    def __init__(self, pc_name, map_point, agg, intensity):
        self.pc_name = pc_name
        self.map_point = map_point
        self.agg_name = agg
        if agg=='avg':
            self.agg_func = np.mean
        elif agg=='std':
            self.agg_func = np.std
        else:
            raise ValueError(f"Aggregation function {agg} not implemented.")
        self.intensity = intensity

    def __repr__(self):
        return f"<{self.__class__.__name__}>{self.agg_name}.{self.intensity}.{self.pc_name}.BIN{self.map_point}"

    def set_config(self, config):
        self.config = config
    
    def set_shape_space(self, df, shapespace):
        self.df = df
        self.space = shapespace
        self.find_available_parameterized_intensities()
        self.load_meshes_and_parameterize()
        
    def find_available_parameterized_intensities(self):
        # Assumes all images have same parameterized intensities
        img_path = self.df.at[self.df.index[0],'PathToRepresentationFile']
        channel_names = AICSImage(img_path).get_channel_names()
        if self.intensity not in channel_names:
            raise ValueError(f"Intensity name {self.intensity} not available in the data.")
        self.intensity_names = channel_names
        
    def read_parameterized_intensity(self, index):
        code = AICSImage(self.df.at[index,'PathToRepresentationFile'])
        code = code.data.squeeze()
        return code
    
    def aggregate_parameterized_intensities(self):
        
        N_CORES = len(os.sched_getaffinity(0))
        with concurrent.futures.ProcessPoolExecutor(N_CORES) as executor:
            pints = list(
                executor.map(self.read_parameterized_intensity, self.df.index))
        pints = np.array(pints)
        
        agg_pint = self.agg_func(pints, axis=0)
        
        channel_id = self.intensity_names.index(self.intensity)
        
        return agg_pint[channel_id]
    
    def load_meshes_and_parameterize(self):
                
        mesh_dna = self.space.get_dna_mesh_of_bin(self.map_point)
        mesh_mem = self.space.get_mem_mesh_of_bin(self.map_point)

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
        
        return domain, coords_param
    
    def morph_parameterized_intensity_on_shape(self, save_as):
        
        agg_pint = self.aggregate_parameterized_intensities()
        
        img = cytoparam.morph_representation_on_shape(
            img=self.domain,
            param_img_coords=self.coords_param,
            representation=agg_pint
        )

        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(img, dimension_order='ZYX', image_name=save_as.stem)
        
        return

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

def create_dataframe_of_celids(df, config):
    prefix = config['aggregation']['aggregate_on']
    pc_names = [f for f in df.columns if prefix in f]
    space = shapespace.ShapeSpace(df[pc_names])
    df_agg = []
    for pc_name in tqdm(pc_names):
        space.set_active_axis(pc_name)
        space.digitize_active_axis()
        for _, intensity in config['parameterization']['intensities']:
            for agg in config['aggregation']['type']:
                for b, _ in space.iter_map_points():
                    indexes = space.get_indexes_in_bin(b)
                    for struct in tqdm(config['structures']['genes'], leave=False):
                        df_struct = df.loc[(df.index.isin(indexes))&(df.structure_name==struct)]
                        if len(df_struct) > 0:
                            df_agg.append({
                                "aggtype": agg,
                                "intensity": intensity,
                                "structure_name": struct,
                                "shapemode": pc_name,
                                "bin": b,
                                "CellIds": df_struct.index.values.tolist()
                            })
                            
    return pd.DataFrame(df_agg)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Batch aggregation.')
    parser.add_argument('--config', help='Path to the JSON config file.', required=True)
    args = vars(parser.parse_args())
    
    with open(args['config'], 'r') as f:
        config = json.load(f)

    df = pd.read_csv(config['csv'], index_col='CellId')
        
    def parse_filename(filename):
        agg, intensity, struct, pc_name, map_point = filename.split('-')
        map_point = int(map_point.split('.')[0].replace('B',''))
        return agg, intensity, struct, pc_name, map_point
    agg, intensity, struct, pc_name, map_point = parse_filename(config['filename'])
    
    space = shapespace.ShapeSpaceBasic()
    space.link_results_folder(config['shapemode_results'])
    space.set_active_axis(pc_name)

    save_as = Path(config['output']) / config['filename']
    Agg = Aggregator(pc_name, map_point, agg, intensity)
    Agg.set_shape_space(df, space)
    Agg.morph_parameterized_intensity_on_shape(save_as)
