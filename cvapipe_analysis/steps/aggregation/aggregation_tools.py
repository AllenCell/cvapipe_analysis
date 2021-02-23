import vtk
import pickle
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

from ..shapemode.avgshape import digitize_shape_mode

def _read_parameterization(row):
    
    """
    Read parameterized intensity representations
    AICSImage.

    Parameters
    --------------------
    df: pd.Series
        Pandas dataframe row that constain the column
        CellRepresentationPath that points to the
        cell representation generated in step
        parameterization.

    Returns
    -------
    code: AICSImage
        AICSImage of the TIF file that represents the
        parameterized intensity representation.
    """
    
    # Delayed Image Reading. Dimension S is kept and
    # used to run over different cell ids. Z=0
    # since the representation is always 2D.
    code = AICSImage(row['CellRepresentationPath'])
    code = code.get_image_dask_data("SCYX", T=0, Z=0)
    
    return code
    
def aggregate_intensity_representations(
    df: pd.DataFrame,
    distributed_executor_address: Optional[str] = None
):

    """
    Aggregate the parameterized intensity representation
    of all cells contained in the input dataframe. All
    cells are processed in parallel via Dask.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains one cell per row and
        the column CellRepresentationPath with the path to
        the corresponding representation.

    Returns
    -------
    agg: dict
        Dict with one key for each representation encoded
        in the images and another key for the aggregation
        type: avg for average and std for standard deviation.
    """
    
    # Process each row
    with DistributedHandler(distributed_executor_address) as handler:
        representations = handler.batched_map(
            _read_parameterization,
            [row for _, row in df.iterrows()]
        )
    # Stack images together
    representations = np.vstack(representations)
    
    # Use dask to compute mean and std
    avg_rep = representations.mean(axis=0).compute()
    std_rep = representations.std(axis=0).compute()

    # Load first image to find channel names
    index = df.index[0]
    img_path = df.at[index,'CellRepresentationPath']
    channel_names = AICSImage(img_path).get_channel_names()
    
    agg = {}
    for ch, ch_name in enumerate(channel_names):
        agg[ch_name] = {
            'avg': avg_rep[ch],
            'std': std_rep[ch]
        }

    return agg

def aggregate_intensities_of_shape_mode(
    df: pd.DataFrame,
    pc_names: List,
    pc_idx: int,
    nbins: int,
    save_dir: Path
):

    """
    Aggregate the representations per srtructure name and
    per bin number. The final aggregation is saved as a
    hyperstack image with format: TC1YX, where T is used
    to store the different bins and C is used to store the
    different structures.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains one cell per row and
        the column CellRepresentationPath with the path to
        the corresponding representation.
    pc_names: List
        List with all features that should be used to
        discretize the PC space (shape space).
    pc_idx: int
        Index of the feature in the list pc_names that will
        be aggregated.
    save_dir: Path
        Path where to save the results.
    nbins: int
        Number of bins that was used to discretize the shape
        space.

    Returns
    -------
    result: List
        List of dict with keys for the pc_name, representation
        name, aggregation type and path to aggregated
        representation.    
    """
    
    # Name of the PC to be processed
    pc_name = pc_names[pc_idx]
        
    # Find the indexes of cells in each bin
    df_filtered, bin_indexes, _, df_freq = digitize_shape_mode(
        df=df,
        feature=pc_name,
        nbins=nbins,
        filter_based_on=pc_names,
        return_freqs_per_structs=True
    )

    # Save dataframe with number of cells
    save_as = save_dir / f"{pc_name}_ncells.csv"
    df_freq.to_csv(save_as)
    
    # Loop over different structures
    agg_struct = {}
    n_structs = len(df_filtered.structure_name.unique())
    for struct, df_struct in tqdm(df_filtered.groupby('structure_name'), total=n_structs):
        # Loop over bins
        agg_bins = {}
        for b in df_filtered.bin.unique():

            df_struct_bin = df_struct.loc[df_struct.bin==b]

            ncells = df_struct_bin.shape[0]

            if ncells > 0:

                agg = aggregate_intensity_representations(df_struct_bin)

                agg_bins[b] = agg
                
        agg_struct[struct] = agg_bins

    # Find all names of the representations in this data
    s = list(agg_struct.keys())[0] # temp structure name
    b = list(agg_struct[s].keys())[0] # temp bin value
    # Every pair (s,b) should share the same names of
    # representation and aggregation types (avg and std).
    rep_names = list(agg_struct[s][b].keys())
    agg_types = list(agg_struct[s][b][rep_names[0]])
            
    return agg_struct, rep_names, agg_types


def load_meshes_and_parameterize(
    pc_name: str,
    bin_number: int,
    df: pd.DataFrame
):

    """
    Load idealized cell and nuclear meshes of a specific bin of
    a specific PC.

    Parameters
    --------------------
    pc_name: str
        Name of the principal component.
    bin_number: int
        Number of the bin along the principal component.
    df: pd.DataFrame
        DataFrame that contains the path to cell and nuclear
        meshes of idealized shapes from shape space. This
        dataframe is produced by the shapemode step.

    Returns
    -------
    domain: np.array
        Numpy array with cell and nucleus voxelization. Cell
        voxels are labeled with 1 and nuclear voxels are labeled
        as 2. Background is 0.
    """
    
    # Use bin number and pc_name to find index of idealized shape in
    # the dataframe
    index = df.loc[(df.shapemode==pc_name) & (df.bin==bin_number)].index
    if len(index) > 1:
        warnings.warn(f"More than one index found for pc {pc_name} and\
        bin {bin_number}. Something seems wrong with the dataframe of\
        VTK paths generated in the step shapemode. Continuing with\
        first index.")
    index = index[0]
    # Load nuclear mesh
    mesh_dna_path = df.at[index, 'dnaMeshPath']
    reader_dna = vtk.vtkPolyDataReader()
    reader_dna.SetFileName(mesh_dna_path)
    reader_dna.Update()
    mesh_dna = reader_dna.GetOutput()
    # Load cell mesh
    mesh_mem_path = df.at[index, 'memMeshPath']
    reader_mem = vtk.vtkPolyDataReader()
    reader_mem.SetFileName(mesh_mem_path)
    reader_mem.Update()
    mesh_mem = reader_mem.GetOutput()

    # Voxelize
    domain, origin = cytoparam.voxelize_meshes([mesh_mem, mesh_dna])

    # Parameterize
    coords_param, _ = cytoparam.parameterize_image_coordinates(
        seg_mem = (domain>0).astype(np.uint8),
        seg_nuc = (domain>1).astype(np.uint8),
        lmax = 16,
        nisos = [32,32]
    )

    return domain, coords_param


def _run_morphing(
    agg_type,
    rep_name,
    bin_number,
    struct,
    agg_structs,
    domain,
    coords
):
    
    '''Wrapper for running cytoparam.morph_representation_on_shape in
    parallel.
    '''
    
    gfp = None
    # Check if the bin is available, meaning cells have been found
    # for this particular bin.
    if bin_number in agg_structs[struct]:
        # Get the current represnetation
        rep = agg_structs[struct][bin_number][rep_name][agg_type]
        # Use cytoparam to morph the aggregated representation into
        # the idealized cell and nuclear shape
        gfp = cytoparam.morph_representation_on_shape(
            img = domain,
            param_img_coords = coords,
            representation = rep
        )
        
    return gfp
    
    
def create_5d_hyperstacks(
    df: pd.DataFrame,
    df_paths: pd.DataFrame,
    pc_names: List,
    pc_idx: int,
    save_dir: Path,
    nbins: int,
    distributed_executor_address: Optional[str]=None
):

    # Aggregate representations: avg and std
    agg_structs, rep_names, agg_types = aggregate_intensities_of_shape_mode(
        df=df,
        pc_names=pc_names,
        pc_idx=pc_idx,
        nbins=nbins,
        save_dir=save_dir
    )
    
    pc_name = pc_names[pc_idx]

    # Save representations in a pickle file
    save_reps_as = save_dir / f"{pc_name}.tif"
    pickle.dump(agg_structs, open(save_rep_as, "wb"))

    # List of structures. This list determines the order in which the
    # channels of the hyperstack are going to be saved. In the future
    # this could come as a parameter extracted from a config file.
    structs = ["FBL", "NPM1", "SON", "SMC1A", "HIST1H2BJ", "LMNB1" ,"NUP153" ,
               "SEC61B", "ATP2A2", "TOMM20", "SLC25A17", "RAB5A", "LAMP1", 
               "ST6GAL1", "CETN2", "TUBA1B", "AAVS1", "ACTB", "ACTN1", "MYH10",
               "GJA1", "TJP1", "DSP", "CTNNB1", "PXN"]
    ns = len(structs)
    
    df_results = []
    # Loop over all representations
    for rep_name in rep_names:
        # Loop over all aggregation types
        for agg_type in agg_types:
            # Loop over bins
            hyperstack = []
            for b in tqdm(range(1,1+nbins)):
                # Parameterize idealized cell and nuclear shape of current bin
                domain, coords_param = load_meshes_and_parameterize(
                    pc_name=pc_name,
                    bin_number=b,
                    df=df_paths
                )
                # Morph average representations into idealized cell and nuclear shape
                with DistributedHandler(distributed_executor_address) as handler:
                    gfps = handler.batched_map(
                        _run_morphing,
                        *[
                            [agg_type] * ns,
                            [rep_name] * ns,
                            [b] * ns,
                            structs,
                            [agg_structs] * ns,
                            [domain] * ns,
                            [coords_param] * ns
                        ]
                    )
                stack = np.zeros((ns,*domain.shape), dtype=np.float32)
                for sid, struct in enumerate(structs):
                    if gfps[sid] is not None:
                        stack[sid] = gfps[sid].copy()
                # Concatenate domain to morphed representations
                stack = np.vstack([stack, domain.reshape(1, *domain.shape)])
                hyperstack.append(stack)

            # Calculate largest czyx bounding box across bins
            shapes = np.array([stack.shape for stack in hyperstack])
            lbb = shapes.max(axis=0)
            
            # Pad all stacks so they end up with similar shapes
            for b in range(nbins):
                # Calculate padding values
                stack_shape = np.array(hyperstack[b].shape)
                # Inferior padding
                pinf = (0.5 * (lbb - stack_shape)).astype(np.int)
                # Superior padding
                psup = lbb - (stack_shape + pinf)
                # Everything into the same list
                pad = [(i,s) for i,s in zip(pinf, psup)]
                # Pad
                hyperstack[b] = np.pad(hyperstack[b], list(pad))

            # Final hyperstack. This variable has ~4Gb for the
            # full hiPS single-cell images dataset.
            hyperstack = np.array(hyperstack)

            # Save hyperstack
            save_as = save_dir / f"{pc_name}_{rep_name}_{agg_type}.tif"
            with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
                writer.save(
                    hyperstack,
                    dimension_order = 'TCZYX',
                    image_name = f"{pc_name}_{rep_name}_{agg_type}",
                    # Add domain to list of structures
                    channel_names = structs + ['domain']
                )
                
            # Store paths
            df_results.append({
                'shapemode': pc_name,
                'aggregation_type': agg_type,
                'scalar': rep_name,
                'hyperstackPath': save_as,
                'representations': save_reps_as
            })

    df_results = pd.DataFrame(df_results)

    return df_results
