import argparse
import concurrent
import pandas as pd
from aicscytoparam import cytoparam

from cvapipe_analysis.tools import io, general, controller

class Parameterizer(io.DataProducer):
    """
    Functionalities for parameterizing channels of
    single cell images.
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.subfolder = 'parameterization/representations'
        
    def workflow(self):

        self.load_single_cell_data()
        self.align_data()

        alias_outer = self.control.get_outer_most_alias_to_parameterize()
        alias_inner = self.control.get_inner_most_alias_to_parameterize()
        
        coeffs_outer, centroid_outer = self.find_shcoeffs_and_centroid(alias_outer)
        coeffs_inner, centroid_inner = self.find_shcoeffs_and_centroid(alias_inner)

        named_imgs = self.get_list_of_imgs_to_create_representation_for()
        
        n = self.control.get_number_of_interpolating_points()
        self.representations = cytoparam.parameterization_from_shcoeffs(
            coeffs_mem = coeffs_outer,
            centroid_mem = centroid_outer,
            coeffs_nuc = coeffs_inner,
            centroid_nuc = centroid_inner,
            nisos = [n, n],
            images_to_probe = named_imgs
        )
        return

    def get_output_file_name(self):
        return f"{self.row.name}.tif"

    def save(self):
        save_as = self.get_output_file_path()
        img = self.representations.get_image_data('CZYX', S=0, T=0)
        self.write_ome_tif(
            save_as, img, channel_names=self.representations.channel_names
        )
        return save_as
    
    def get_list_of_imgs_to_create_representation_for(self):
        named_imgs = []
        for alias in self.control.get_aliases_to_parameterize():
            channel_name = self.control.get_channel_from_alias(alias)
            ch = self.channels.index(channel_name)
            named_imgs.append((alias, self.data_aligned[ch]))
        return named_imgs

    def find_shcoeffs_and_centroid(self, alias):
        coeffs = dict(
            (f"{k.replace(f'{alias}_','').replace('_lcc','')}",v)
            for k, v in self.row.items() if f'{alias}_shcoeffs_L' in k
        )
        centroid = [
            self.row[f'{alias}_transform_{r}c_lcc']
            for r in ['x', 'y', 'z']
        ]
        return coeffs, centroid
    
if __name__ == "__main__":

    config = general.load_config_file()
    control = controller.Controller(config)

    parser = argparse.ArgumentParser(description='Batch single cell parameterization.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'], index_col='CellId')
    print(f"Processing dataframe of shape {df.shape}")

    parameterizer = Parameterizer(control)
    with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
        executor.map(parameterizer.execute, [row for _,row in df.iterrows()])