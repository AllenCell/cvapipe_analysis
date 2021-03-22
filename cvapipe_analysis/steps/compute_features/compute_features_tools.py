import os
import json
import argparse
import concurrent
import numpy as np
import pandas as pd
from pathlib import Path
from aicsshparam import shtools, shparam
from skimage import measure as skmeasure
from skimage import morphology as skmorpho

from cvapipe_analysis.tools import general, cluster

class FeatureCalculator(general.DataProducer):
    """
    Class for feature extraction.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """
    
    subfolder = 'computefeatures/cell_features'
    
    def __init__(self, config):
        super().__init__(config)
    
    def save(self):
        save_as = self.get_rel_output_file_path_as_str(self.row)
        df = pd.DataFrame([self.features])
        df.index = df.index.rename('CellId')
        df.to_csv(save_as)
        return save_as

    def get_features_from_binary_image(self, input_image, input_reference_image, compute_shcoeffs=True):
        features = {}
        input_image = (input_image>0).astype(np.uint8)
        input_image_lcc = skmeasure.label(input_image)
        # Number of connected components
        features[f'connectivity_cc'] = input_image_lcc.max()
        if features[f'connectivity_cc'] > 0:

            # Find largest connected component (lcc)
            counts = np.bincount(input_image_lcc.reshape(-1))
            lcc = 1+np.argmax(counts[1:])
            input_image_lcc[input_image_lcc!=lcc] = 0
            input_image_lcc[input_image_lcc==lcc] = 1
            input_image_lcc = input_image_lcc.astype(np.uint8)

            for img, suffix in zip([input_image, input_image_lcc], ['', '_lcc']):
                z, y, x = np.where(img)
                features[f'shape_volume{suffix}'] = img.sum()
                features[f'position_depth{suffix}'] = 1+np.ptp(z)
                for uname, u in zip(['x', 'y', 'z'], [x, y, z]):
                    features[f'position_{uname}_centroid{suffix}'] = u.mean()
                features[f'roundness_surface_area{suffix}'] = self.get_surface_area(img)
        else:
            # If no foreground pixels are found
            for img, suffix in zip([input_image,input_image_lcc], ['', '_lcc']):
                features[f'shape_volume{suffix}'] = np.nan
                features[f'position_depth{suffix}'] = np.nan
                for uname in ['x', 'y', 'z']:
                    features[f'position_{uname}_centroid{suffix}'] = np.nan
                features[f'roundness_surface_area{suffix}'] = np.nan

        if not compute_shcoeffs:
            return features

        angle = np.nan
        if input_reference_image is not None:
            # Get alignment angle based on the reference image. Variance
            # paper uses make_unique = False
            input_ref_image_aligned, angle = shtools.align_image_2d(
                image=input_reference_image,
                make_unique=self.config['data']['align']['unique']
            )
            # Rotate input image according the reference alignment angle
            input_image_lcc_aligned = shtools.apply_image_alignment_2d(
                image=input_image_lcc,
                angle=angle
            ).squeeze()
        else:
            input_image_lcc_aligned = input_image_lcc

        (coeffs, _), (_, _, _, transform) = shparam.get_shcoeffs(
            image=input_image_lcc_aligned,
            lmax=self.config['parameterization']['lmax'],
            sigma=self.config['parameterization']['sigma'],
            alignment_2d=False
        )

        if transform is not None:
            transform = {
                'shcoeffs_transform_xc_lcc': transform[0],
                'shcoeffs_transform_yc_lcc': transform[1],
                'shcoeffs_transform_zc_lcc': transform[2],
                'shcoeffs_transform_angle_lcc': angle,
            }
        else:
            transform = {
                'shcoeffs_transform_xc_lcc': np.nan,
                'shcoeffs_transform_yc_lcc': np.nan,
                'shcoeffs_transform_zc_lcc': np.nan,
                'shcoeffs_transform_angle_lcc': np.nan,
            }

        # Add suffix to identify coeffs have been calculated on the
        # largest connected component
        coeffs = dict(
            (f'{key}_lcc',value) for (key,value) in coeffs.items()
        )
        features.update(coeffs)
        features.update(transform)
        return features
    
    def get_reference_image_for_alignment(self, segs):
        '''Returns None if no alignment reference is specified.'''
        align_reference = self.config['data']['align']['reference']
        if isinstance(align_reference, str):
            return segs[self.config['data']['segmentation'][align_reference]['name']]
        return None

    def workflow(self, row):
        self.row = row
        reader = general.LocalStagingReader(self.config, row)
        segs = reader.get_single_cell_images('crop_seg')
        raws = reader.get_single_cell_images('crop_raw')
        
        features={}
        align_reference_img = self.get_reference_image_for_alignment(segs)
        for obj, channel in self.config['data']['segmentation'].items():
            if isinstance(channel, dict):
                features_obj = self.get_features_from_binary_image(
                    input_image=segs[channel['name']],
                    input_reference_image=align_reference_img,
                    compute_shcoeffs=not(obj=='structure')
                )
                features_obj = dict(
                    (f"{channel['alias']}_{key}", value) for (key, value) in features_obj.items()
                )
                features.update(features_obj)
        self.features=pd.Series(features, name=self.row.name)
        return
    
    @staticmethod
    def get_output_file_name(row):
        return f"{row.name}.csv"

    @staticmethod
    def get_surface_area(input_img):
        # Forces a 1 pixel-wide offset to avoid problems with binary
        # erosion algorithm
        input_img[:, :, [0, -1]] = 0
        input_img[:, [0, -1], :] = 0
        input_img[[0, -1], :, :] = 0
        input_img_surface = np.logical_xor(input_img, skmorpho.binary_erosion(input_img)).astype(np.uint8)
        
        # Loop through the boundary voxels to calculate the number of
        # boundary faces. Using 6-neighborhod.
        pxl_z, pxl_y, pxl_x = np.nonzero(input_img_surface)
        dx = np.array([ 0, -1,  0,  1,  0,  0])
        dy = np.array([ 0,  0,  1,  0, -1,  0])
        dz = np.array([-1,  0,  0,  0,  0,  1])

        surface_area = 0
        for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
            surface_area += 6-input_img_surface[k+dz, j+dy, i+dx].sum()
        return int(surface_area)

if __name__ == "__main__":
    
    config = general.load_config_file()
    
    parser = argparse.ArgumentParser(description='Batch single cell feature extraction.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'], index_col='CellId')
    print(f"Processing dataframe of shape {df.shape}")

    calculator = FeatureCalculator(config)
    with concurrent.futures.ProcessPoolExecutor(cluster.get_ncores()) as executor:
        executor.map(calculator.execute, [row for _,row in df.iterrows()])
    
