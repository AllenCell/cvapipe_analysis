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

from cvapipe_analysis.tools import io, general, controller

class FeatureCalculator(io.DataProducer):
    """
    Class for feature extraction.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """

    def __init__(self, control):
        super().__init__(control)
        self.subfolder = 'computefeatures/cell_features'

    def workflow(self):
        device = io.LocalStagingIO(self.control)
        self.segs = device.get_single_cell_images(self.row)
        features = {}
        for alias, channel in self.control.get_data_seg_alias_channel_dict().items():
            features_alias = self.get_features_from_binary_image(alias)
            features_alias = dict(
                (f"{alias}_{k}", v) for (k, v) in features_alias.items()
            )
            features.update(features_alias)
        self.features = pd.Series(features, name=self.row.name)
        return

    def get_output_file_name(self):
        return f"{self.row.name}.csv"
    
    def save(self):
        save_as = self.get_output_file_path()
        df = pd.DataFrame([self.features])
        df.index = df.index.rename('CellId')
        df.to_csv(save_as)
        return save_as

    def get_features_from_binary_image(self, alias):
        features = {}
        ch = self.control.get_channel_from_alias(alias)
        input_image = self.segs[ch]
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

        if not self.control.should_calculate_shcoeffs(alias):
            return features
        
        input_image_lcc_aligned, angle = self.align_if_necessary(
            self.segs, self.control, input_image_lcc)
        
        (coeffs, _), (_, _, _, transform) = shparam.get_shcoeffs(
            image=input_image_lcc_aligned,
            lmax=self.control.get_lmax(),
            sigma=self.control.get_sigma(),
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
            (f"{key}_lcc", value) for (key, value) in coeffs.items()
        )
        features.update(coeffs)
        features.update(transform)
        return features

    
    @staticmethod
    def align_if_necessary(ref, control, imgs=None, channel_names=None):
        angle = np.nan
        if control.should_align():
            alias = control.get_alignment_reference_alias()
            if alias is None:
                raise ValueError("Specify an reference alias for alignment.")
            ch = control.get_channel_from_alias(alias)
            unique = control.make_alignment_unique()
            if isinstance(ref, dict):
                if ch not in ref:
                    raise ValueError(f"Channel {ch} not found.")
            else:
                if channel_names is None:
                    raise TypeError("Provide channel names when ref is not a dict.")
                ch = channel_names.index(ch)
            if imgs is None:
                imgs = ref.copy()
            ref = ref[ch]
            imgs_aligned, angle = FeatureCalculator.run_alignment_with_reference(
                ref, unique, imgs
            )
        return imgs_aligned, angle
    
    @staticmethod
    def run_alignment_with_reference(ref, unique, img):
        if ref.ndim != 3:
            raise ValueError("Dim of ref image should be 3.")
        _, angle = shtools.align_image_2d(ref, unique)
        img_aligned = shtools.apply_image_alignment_2d(img, angle)
        if img.ndim == 3:
            img_aligned = img_aligned.squeeze()
        return img_aligned, angle
    
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
        dx = np.array([0, -1, 0, 1, 0, 0])
        dy = np.array([0, 0, 1, 0, -1, 0])
        dz = np.array([-1, 0, 0, 0, 0, 1])
        surface_area = 0
        for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
            surface_area += 6 - input_img_surface[k+dz, j+dy, i+dx].sum()
        return int(surface_area)

if __name__ == "__main__":

    config = general.load_config_file()
    control = controller.Controller(config)

    parser = argparse.ArgumentParser(description="Batch single cell feature extraction.")
    parser.add_argument("--csv", help="Path to the dataframe.", required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args["csv"], index_col="CellId")
    print(f"Processing dataframe of shape {df.shape}")

    calculator = FeatureCalculator(control)
    with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
        executor.map(calculator.execute, [row for _, row in df.iterrows()])

