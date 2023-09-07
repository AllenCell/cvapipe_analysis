import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicscytoparam import cytoparam
import concurrent

from cvapipe_analysis.tools import io, general, controller

class Aggregator(io.DataProducer):
    """
    The goal of this class is to have a combination of
    parameters as input, including some CellIds. The
    corresponding cells have their parameterized intensity
    representation morphed into the appropriated shape
    space shape according to the input parameters.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files away from the
    places their are saved.
    """

    def __init__(self, control):
        super().__init__(control)
        self.subfolder = 'aggregation/aggmorph'

    def workflow(self):
        self.set_agg_function()
        self.print("Aggregating representations...")
        self.aggregate_parameterized_intensities()
        self.print("Generating average morphed cells...")
        self.morph_on_shapemode_shape()

    def get_output_file_name(self):
        return f"{self.get_prefix_from_row(self.row)}.tif"

    def save(self):
        # TODO: update aicsimageio and use TIF metadata for channel names
        n = len(self.CellIds)
        save_as = self.get_output_file_path()
        # Save morphed cell
        img = self.morphed
        self.write_ome_tif(save_as, img, ["domain", self.row.alias], f"N{n}")
        # Save agg representation
        img = self.aggregated_parameterized_intensity
        save_as = Path(str(save_as).replace('aggmorph', 'repsagg'))
        self.write_ome_tif(save_as, img, self.row.alias, f"N{n}")
        # Save norm agg representation
        img = self.aggregated_norm_parameterized_intensity
        save_as = Path(str(save_as).replace('.tif', '_norm.tif'))
        self.write_ome_tif(save_as, img, self.row.alias, f"N{n}")
        return save_as

    def set_shape_space(self, space):
        self.space = space
        self.load_parameterization_manifest()

    def aggregate_parameterized_intensities(self):
        nc = self.control.get_ncores()
        ncells = len(self.CellIds)
        self.print(f"\t{ncells} cells found for {self.row.structure}")
        if not ncells:
            raise ValueError("No cells found for parameterization.")
        self.print(f"\tLoading PILRs...")
        with concurrent.futures.ProcessPoolExecutor(nc) as executor:
            pints = list(executor.map(self.read_parameterized_intensity, self.CellIds))
        pints = np.array([p for p in pints if p is not None])
        self.print(f"\tAggregating...")
        pints_norm = self.normalize_representations(pints)
        agg_pint = self.agg_func(pints, axis=0)
        agg_pint_norm = self.agg_func(pints_norm, axis=0)
        # Selecting the channel according to the alias of interest
        ch = self.control.get_aliases_to_parameterize().index(self.row.alias)
        agg_pint = np.expand_dims(agg_pint[ch], 0)
        agg_pint_norm = np.expand_dims(agg_pint_norm[ch], 0)
        self.aggregated_parameterized_intensity = agg_pint
        self.aggregated_norm_parameterized_intensity = agg_pint_norm
        return

    def set_agg_function(self):
        if self.row.aggtype == 'avg':
            self.agg_func = np.mean
        elif self.row.aggtype == 'std':
            self.agg_func = np.std
        else:
            raise ValueError(f"Aggregation type {self.row.aggtype} is not implemented.")

    def voxelize_and_parameterize_shapemode_shape(self):
        n = self.control.get_number_of_interpolating_points()
        alias_outer = self.control.get_outer_most_alias_to_parameterize()
        alias_inner = self.control.get_inner_most_alias_to_parameterize()
        reader = self.read_map_point_mesh
        if self.row.shape_mode == "NdSphere":
            # Change the reader function in case we are aggregating
            # cells inside the NdSphere
            reader = self.read_mean_shape_mesh
        mesh_outer = reader(alias_outer)
        mesh_inner = reader(alias_inner)
        domain, origin = cytoparam.voxelize_meshes([mesh_outer, mesh_inner])
        coords_param, _ = cytoparam.parameterize_image_coordinates(
            seg_mem=(domain>0).astype(np.uint8),
            seg_nuc=(domain>1).astype(np.uint8),
            lmax=self.control.get_lmax(), nisos=[n, n]
        )
        self.domain = domain
        self.origin = origin
        self.coords_param = coords_param
        return

    def morph_on_shapemode_shape(self):
        self.print("\tVoxelizing shape mode...")
        self.voxelize_and_parameterize_shapemode_shape()
        self.print("\tMorphing...")
        self.morphed = cytoparam.morph_representation_on_shape(
            img=self.domain,
            param_img_coords=self.coords_param,
            representation=self.aggregated_parameterized_intensity
        )
        self.morphed = np.stack([self.domain, self.morphed])
        return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch single cell feature extraction.")
    parser.add_argument("--staging", help="Path to staging.", required=True)
    parser.add_argument("--csv", help="Path to the dataframe.", required=True)
    args = vars(parser.parse_args())

    config = general.load_config_file(args["staging"])
    control = controller.Controller(config)

    df = pd.read_csv(args['csv'], index_col=0)

    aggregator = Aggregator(control)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        '''Concurrent processes inside. Do not use concurrent here.'''
        aggregator.execute(row)
