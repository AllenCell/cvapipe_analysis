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
