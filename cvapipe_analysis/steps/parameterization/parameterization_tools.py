from aicsshparam import shtools
from aicscytoparam import cytoparam

from ..compute_features.compute_features_tools import get_segmentations

def parameterize(data_folder, row):
    
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
    channels = row['name_dict']
    _, _, seg_str = get_segmentations(
        folder = data_folder,
        path_to_seg = row['crop_seg'],
        channels = eval(channels)['crop_seg']
    )

    # Rotate structure channel with same angle used to align the cell
    seg_str = shtools.apply_image_alignment_2d(
        image = seg_str,
        angle = row['mem_shcoeffs_transform_angle_lcc']
    ).squeeze()

    # Find cell coefficients and centroid. Also rename coeffs to agree
    # with aics-cytoparam
    coeffs_mem = dict(
        (f"{k.replace('mem_shcoeffs_','').replace('_lcc','')}",v)
        for k,v in row.items() if 'mem_shcoeffs_L' in k
    )
    centroid_mem = [
        row[f'mem_shcoeffs_transform_{r}c_lcc'] for r in ['x','y','z']
    ]
    # Find nuclear coefficients and centroid. Also rename coeffs to agree
    # with aics-cytoparam
    coeffs_dna = dict(
        (f"{k.replace('dna_shcoeffs_','').replace('_lcc','')}",v)
        for k,v in row.items() if 'dna_shcoeffs_L' in k
    )
    centroid_dna = [
        row[f'dna_shcoeffs_transform_{r}c_lcc'] for r in ['x','y','z']
    ]    
    
    # Run aics-cytoparam
    _, _ = cytoparam.parameterization_from_shcoeffs(
        coeffs_mem = coeffs_mem,
        centroid_mem = centroid_mem,
        coeffs_nuc = coeffs_dna,
        centroid_nuc = centroid_dna,
        nisos = [32,32],
        images_to_probe = [
            ('structure_segmentation', seg_str)
        ]
    )
    
    return None