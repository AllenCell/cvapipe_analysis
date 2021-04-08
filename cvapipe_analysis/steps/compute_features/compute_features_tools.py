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
        self.subfolder = "computefeatures/cell_features"

    def workflow(self):
        self.load_single_cell_data()
        self.align_data()
        self.compute_all_features()
        return

    def get_output_file_name(self):
        return f"{self.row.name}.csv"
    
    def save(self):
        save_as = self.get_output_file_path()
        df = pd.DataFrame([self.features])
        df.index = df.index.rename('CellId')
        df.to_csv(save_as)
        return save_as
    
    def compute_all_features(self):
        features = {}
        for alias in self.control.get_aliases_for_feature_extraction():
            fs = self.compute_features_for_alias(alias)
            fs = dict((f"{alias}_{k}", v) for k, v in fs.items())
            features.update(fs)
        self.features = pd.Series(features, name=self.row.name)
        return
    
    def compute_features_for_alias(self, alias):
        channel = self.control.get_channel_from_alias(alias)
        chId = self.channels.index(channel)
        features = self.get_basic_features(self.data[chId])
        if self.control.should_calculate_shcoeffs(alias):
            coeffs = self.get_coeff_features(self.data_aligned[chId])
            features.update(coeffs)
        return features
    
    def get_basic_features(self, img):
        features = {}
        input_image = img.copy()
        input_image = (input_image>0).astype(np.uint8)
        input_image_lcc = skmeasure.label(input_image)
        features[f'connectivity_cc'] = input_image_lcc.max()
        if features[f'connectivity_cc'] > 0:
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
            for img, suffix in zip([input_image,input_image_lcc], ['', '_lcc']):
                features[f'shape_volume{suffix}'] = np.nan
                features[f'position_depth{suffix}'] = np.nan
                for uname in ['x', 'y', 'z']:
                    features[f'position_{uname}_centroid{suffix}'] = np.nan
                features[f'roundness_surface_area{suffix}'] = np.nan
        return features

    def get_coeff_features(self, img):
        (coeffs, _), (_, _, _, transform) = shparam.get_shcoeffs(
            image=img,
            lmax=self.control.get_lmax(),
            sigma=self.control.get_sigma(),
            alignment_2d=False
        )
        coeffs = dict((f"{k}_lcc", v) for k, v in coeffs.items())
        transform = {
            'transform_xc_lcc': np.nan if transform is None else transform[0],
            'transform_yc_lcc': np.nan if transform is None else transform[1],
            'transform_zc_lcc': np.nan if transform is None else transform[2]
        }
        transform['transform_angle_lcc'] = self.angle
        coeffs.update(transform)
        return coeffs
    
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

