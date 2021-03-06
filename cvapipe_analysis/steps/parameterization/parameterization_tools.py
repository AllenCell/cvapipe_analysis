import os
import json
import argparse
import concurrent
from pathlib import Path
from aicsimageio import writers
from aicsshparam import shtools
from aicscytoparam import cytoparam

from cvapipe_analysis.tools import general

def parameterize(data_folder, row, save_as):
    
    """
    Calculates the surface area of a binary shape by counting the
    number of boundary faces.

    Parameters
    --------------------
    input_image: ndarray
        3D input image representing the single cell segmentation that
        represents either dna, cell or structure.

    Returns
    -------
    result: int
        Number of boundary faces.
    """

    # Load correct segmentations
    channels = eval(row.name_dict)
    path_seg = data_folder / row.crop_seg
    _, _, seg_str = general.get_segmentations(
        path_seg=path_seg,
        channels=channels['crop_seg']
    )

    # Load FP image
    path_raw = data_folder / row.crop_raw
    _, _, struct = general.get_raws(
        path_raw=path_raw,
        channels=channels['crop_raw']
    )
        
    # Rotate structure segmentation with same angle used to align the cell
    seg_str = shtools.apply_image_alignment_2d(
        image = seg_str,
        angle = row.mem_shcoeffs_transform_angle_lcc
    ).squeeze()

    # Rotate FP structure with same angle used to align the cell
    seg_str = shtools.apply_image_alignment_2d(
        image = struct,
        angle = row.mem_shcoeffs_transform_angle_lcc
    ).squeeze()

    # Find cell coefficients and centroid. Also rename coeffs to agree
    # with aics-cytoparam
    coeffs_mem = dict(
        (f"{k.replace('mem_','').replace('_lcc','')}",v)
        for k,v in row.items() if 'mem_shcoeffs_L' in k
    )
    centroid_mem = [
        row[f'mem_shcoeffs_transform_{r}c_lcc'] for r in ['x','y','z']
    ]
    # Find nuclear coefficients and centroid. Also rename coeffs to agree
    # with aics-cytoparam
    coeffs_dna = dict(
        (f"{k.replace('dna_','').replace('_lcc','')}",v)
        for k,v in row.items() if 'dna_shcoeffs_L' in k
    )
    centroid_dna = [
        row[f'dna_shcoeffs_transform_{r}c_lcc'] for r in ['x','y','z']
    ]    
    
    # Run aics-cytoparam
    representations = cytoparam.parameterization_from_shcoeffs(
        coeffs_mem = coeffs_mem,
        centroid_mem = centroid_mem,
        coeffs_nuc = coeffs_dna,
        centroid_nuc = centroid_dna,
        nisos = [32,32],
        images_to_probe = [
            ('structure', struct),
            ('structure_seg', seg_str)
        ]
    )
    
    # Save representation as TIFF file
    with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
        writer.save(
            representations.get_image_data('CZYX', S=0, T=0),
            dimension_order = 'CZYX',
            image_name = save_as.stem,
            channel_names = representations.channel_names
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Batch single cell parameterization.')
    parser.add_argument('--config', help='Path to the JSON config file.', required=True)
    args = vars(parser.parse_args())
    
    with open(args['config'], 'r') as f:
        config = json.load(f)
    
    df = general.read_chunk_of_dataframe(config)
        
    print(f"Processing dataframe of shape {df.shape}")
        
    def wrapper_for_parameterization(index):
        
        row = df.loc[index]
        data_folder = Path(config['data_folder'])
        path_output = Path(config['output']) / f"{index}.tif"

        try:
            parameterize(data_folder, row, path_output)
            print(f"Index {index} complete.")
        except:
            print(f"Index {index} FAILED.")
            
    N_CORES = len(os.sched_getaffinity(0))
    with concurrent.futures.ProcessPoolExecutor(N_CORES) as executor:
        executor.map(wrapper_for_parameterization, df.index)

