import os
import json
import argparse
import concurrent
import pandas as pd
from pathlib import Path
from aicsimageio import writers
from aicsshparam import shtools
from aicscytoparam import cytoparam

from cvapipe_analysis.tools import general

class Parameterizer(general.DataProducer):
    """
    Desc
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    
    subfolder = 'parameterization/representations'
    
    def __init__(self, config):
        super().__init__(config)
            
    def set_row(self, row):
        self.row = row
        
    def parameterize(self):
        channels = eval(self.row.name_dict)
        path_seg = self.abs_path_local_staging/f"loaddata/{self.row.crop_seg}"
        _, _, seg_str = general.get_segmentations(
            path_seg=path_seg,
            channels=channels['crop_seg']
        )
        path_raw = self.abs_path_local_staging/f"loaddata/{self.row.crop_raw}"
        _, _, raw_str = general.get_raws(
            path_raw=path_raw,
            channels=channels['crop_raw']
        )

        # Rotate structure segmentation with same angle used to align the cell
        self.seg_str_aligned = shtools.apply_image_alignment_2d(
            image = seg_str,
            angle = self.row.mem_shcoeffs_transform_angle_lcc
        ).squeeze()
        # Rotate FP structure with same angle used to align the cell
        self.raw_str_aligned = shtools.apply_image_alignment_2d(
            image = raw_str,
            angle = self.row.mem_shcoeffs_transform_angle_lcc
        ).squeeze()

        # Find cell coefficients and centroid. Also rename coeffs to agree
        # with aics-cytoparam
        coeffs_mem = dict(
            (f"{k.replace('mem_','').replace('_lcc','')}",v)
            for k,v in self.row.items() if 'mem_shcoeffs_L' in k
        )
        centroid_mem = [
            self.row[f'mem_shcoeffs_transform_{r}c_lcc'] for r in ['x','y','z']
        ]
        # Find nuclear coefficients and centroid. Also rename coeffs to agree
        # with aics-cytoparam
        coeffs_dna = dict(
            (f"{k.replace('dna_','').replace('_lcc','')}",v)
            for k,v in self.row.items() if 'dna_shcoeffs_L' in k
        )
        centroid_dna = [
            self.row[f'dna_shcoeffs_transform_{r}c_lcc'] for r in ['x','y','z']
        ]    

        # Run aics-cytoparam
        self.representations = cytoparam.parameterization_from_shcoeffs(
            coeffs_mem = coeffs_mem,
            centroid_mem = centroid_mem,
            coeffs_nuc = coeffs_dna,
            centroid_nuc = centroid_dna,
            nisos = [32,32],
            # The names below should come from the config file
            # once the feature calculation is implemented
            # as a class.
            images_to_probe = [
                ('GFP', self.raw_str_aligned),
                ('SEG', self.seg_str_aligned)
            ]
        )
    
    @staticmethod
    def get_output_file_name(row):
        return f"{row.name}.tif"
    
    def save(self):
        save_as = self.get_rel_output_file_path_as_str(self.row)
        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(
                self.representations.get_image_data('CZYX', S=0, T=0),
                dimension_order = 'CZYX',
                image_name = Path(save_as).stem,
                channel_names = self.representations.channel_names
            )
        return save_as

    def workflow(self, row):
        rel_path_to_output_file = self.check_output_exist(row)
        if (rel_path_to_output_file is None) or self.config['project']['overwrite']:
            self.set_row(row)
            try:
                self.parameterize()
                self.save()
            except:
                rel_path_to_output_file = None
        self.status(row.name, rel_path_to_output_file)
        return rel_path_to_output_file
    
if __name__ == "__main__":

    config = general.load_config_file()
    
    parser = argparse.ArgumentParser(description='Batch single cell parameterization.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'], index_col='CellId')
    print(f"Processing dataframe of shape {df.shape}")

    parameterizer = Parameterizer(config)
    N_CORES = len(os.sched_getaffinity(0))
    with concurrent.futures.ProcessPoolExecutor(N_CORES) as executor:
        executor.map(
            parameterizer.workflow, [row for _,row in df.iterrows()]
        )