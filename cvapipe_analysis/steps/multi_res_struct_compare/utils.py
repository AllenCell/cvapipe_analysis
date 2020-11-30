from pathlib import Path

import numpy as np
import pandas as pd
import logging

from scipy.stats import pearsonr, gmean
from skimage.measure import block_reduce
from skimage.util import crop
from scipy.special import comb

# from typing import List, Union, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from aicsimageio import AICSImage

from .constants import (
    DatasetFieldsMorphed,
    DatasetFieldsIC,
    DatasetFieldsAverageMorphed,
    StructureGenes,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def make_5d_stack_cross_corr_dataframe(num_structs, num_bins):

    N_Pairs = comb(num_structs, 2, exact=True)

    channel_pairs = draw_pairs(np.arange(num_structs), n_pairs=N_Pairs)

    # Make dataset of cross correlation indices
    dataset = pd.DataFrame()

    # Structure Genes
    GeneOrder = list(StructureGenes.__dict__.values())[1:-3]

    for bins in range(num_bins):
        this_dataset = pd.DataFrame(
            channel_pairs, columns=["ChannelIndex_i", "ChannelIndex_j"]
        )
        this_dataset["PC_bin"] = bins
        this_dataset["ChannelGeneName_i"] = this_dataset["ChannelIndex_i"].apply(
            lambda x: GeneOrder[x]
        )
        this_dataset["ChannelGeneName_j"] = this_dataset["ChannelIndex_j"].apply(
            lambda x: GeneOrder[x]
        )
        dataset = dataset.append(this_dataset)

    return dataset


def blockreduce_pyramid(input_arr, block_size=(2, 2, 2), func=np.max, max_iters=12):
    """
    Parameters
        ----------
        input_arr: np.array
            Input array to iteratively downsample
            Default: Path("local_staging/singlecellimages/manifest.csv")
        block_size: Tuple(int)
            Block size for iterative array reduction.  All voxels in this block
            are merged via func into one voxel during the downsample.
            Default: (2, 2, 2)
        func: Callable[[np.array], float]
            Function to apply to block_size voxels to merge them into one new voxel.
            Default: np.max
        max_iters: int
            Maximum number of downsampling rounds before ending at a one voxel cell.
            Default: 12
        Returns
        -------
        result: Dict[float, np.array]
            Dictionary of reduced arrays.
            Keys are reduction fold, values the reduced array.
    """

    # how much are we downsampling per round
    fold = gmean(block_size)

    # original image
    i = 0
    pyramid = {fold ** i: input_arr.copy()}

    # downsample and save to dict
    i = 1
    while (i <= max_iters) and (np.max(pyramid[fold ** (i - 1)].shape) > 1):
        pyramid[fold ** i] = block_reduce(pyramid[fold ** (i - 1)], block_size, func)
        i += 1

    return pyramid


def safe_pearsonr(arr1, arr2):
    """Sensibly handle degenerate cases."""
    assert arr1.shape == arr2.shape

    imgs_same = np.all(arr1 == arr2)
    stdv_1_zero = len(np.unique(arr1)) == 1
    stdv_2_zero = len(np.unique(arr2)) == 1

    if (stdv_1_zero | stdv_2_zero) & imgs_same:
        corr = 1.0
    elif (stdv_1_zero | stdv_2_zero) & (not imgs_same):
        corr = 0.0
    else:
        corr, _ = pearsonr(arr1, arr2)

    return corr


def pyramid_correlation(
    img1, img2, mask1=None, mask2=None, permute=False, **pyramid_kwargs
):
    # make sure inputs are all the same shape
    assert img1.shape == img2.shape
    if mask1 is None:
        mask1 = np.ones_like(img1)
    assert mask1.shape == img1.shape

    if mask2 is None:
        mask2 = np.ones_like(img2)
    assert mask2.shape == img2.shape

    # make image pyramids
    pyramid_1 = blockreduce_pyramid(img1, **pyramid_kwargs)
    pyramid_2 = blockreduce_pyramid(img2, **pyramid_kwargs)

    # also make a mask pyramid
    mask_kwargs = pyramid_kwargs.copy()
    mask_kwargs["func"] = np.max

    pyramid_mask_1 = blockreduce_pyramid(mask1, **mask_kwargs)

    if (mask1 == mask2).all():
        pyramid_mask_2 = pyramid_mask_1
    else:
        pyramid_mask_2 = blockreduce_pyramid(mask2, **mask_kwargs)

    # make sure everything has the same keys
    assert pyramid_1.keys() == pyramid_2.keys()
    assert pyramid_mask_1.keys() == pyramid_1.keys()
    assert pyramid_mask_2.keys() == pyramid_2.keys()

    pyramid_1_masked_flat = {k: v.flatten() for k, v in pyramid_1.items()}
    pyramid_2_masked_flat = {k: v.flatten() for k, v in pyramid_2.items()}

    # at each resolution, find corr
    if not permute:
        corrs = {
            k: safe_pearsonr(pyramid_1_masked_flat[k], pyramid_2_masked_flat[k])
            for k in sorted(
                set({**pyramid_1_masked_flat, **pyramid_2_masked_flat}.keys())
            )
        }
    else:
        # shuffle voxels in one pyramid if we want the permuted baseline
        pyramid_1_masked_flat_permuted = pyramid_1_masked_flat.copy()
        for k in pyramid_1_masked_flat_permuted.keys():
            np.random.shuffle(pyramid_1_masked_flat[k])
        corrs = {
            k: safe_pearsonr(
                pyramid_1_masked_flat_permuted[k], pyramid_2_masked_flat[k]
            )
            for k in sorted(
                set({**pyramid_1_masked_flat_permuted, **pyramid_2_masked_flat}.keys())
            )
        }

    return corrs


def get_cell_mask(image_path, crop_size=(64, 160, 96), cell_mask_channel_ind=1):
    """
    Take a path to a tiff and return the masked gfp 3d volume
    """

    # load image
    image = AICSImage(image_path)
    data_6d = image.data
    mask_3d = data_6d[0, 0, cell_mask_channel_ind, :, :, :]

    # crop to desired shape
    z_dim, y_dim, x_dim = mask_3d.shape
    z_desired, y_desired, x_desired = crop_size
    z_crop = (z_dim - z_desired) // 2
    y_crop = (y_dim - y_desired) // 2
    x_crop = (x_dim - x_desired) // 2
    mask_3d = crop(mask_3d, ((z_crop, z_crop), (y_crop, y_crop), (x_crop, x_crop)))
    assert mask_3d.shape == crop_size

    return mask_3d


def get_gfp_single_channel_img(image_path, crop_size=(64, 160, 96), gfp_channel_ind=0):
    """
    Take a path to a tiff and return the masked gfp 3d volume
    """

    # load image
    image = AICSImage(image_path)
    data_6d = image.data
    gfp_3d = data_6d[0, 0, gfp_channel_ind, :, :, :]

    # crop to desired shape
    z_dim, y_dim, x_dim = gfp_3d.shape
    z_desired, y_desired, x_desired = crop_size
    z_crop = (z_dim - z_desired) // 2
    y_crop = (y_dim - y_desired) // 2
    x_crop = (x_dim - x_desired) // 2
    gfp_3d = crop(gfp_3d, ((z_crop, z_crop), (y_crop, y_crop), (x_crop, x_crop)))
    assert gfp_3d.shape == crop_size

    return gfp_3d


def draw_pairs(input_list, n_pairs=1):
    """
    Draw unique (ordered) pairs of examples from input_list at random.
    Input list is not a list of pairs, just a list of single exemplars.
    Example:
        >>> draw_pairs([0,1,2,3], n_pairs=3)
        >>> {(1,2), (2,3), (0,3)}
    Note:
        A pair is only unique up to order, e.g. (1,2) == (2,1).  this function
        only returns and compared sorted tuple to handle this
    """

    # make sure requested number of uniquepairs in possible
    L = len(input_list)
    assert n_pairs <= L * (L - 1) / 2

    # draw n_pairs of size 2 sets from input_list
    pairs = set()
    while len(pairs) < n_pairs:
        pairs |= {frozenset(sorted(np.random.choice(input_list, 2, replace=False)))}

    # return a set of ordered tuples to not weird people out
    return {tuple(sorted(p)) for p in pairs}


def pct_normalization_and_8bit(raw, pct_range=[50, 99]):
    msk = raw > 0
    values = raw[msk]
    if len(values):
        pcts = np.percentile(values, pct_range)
        if pcts[1] > pcts[0]:
            values = np.clip(values, *pcts)
            values = (values - pcts[0]) / (pcts[1] - pcts[0])
            values = np.clip(values, 0, 1)
            raw[msk] = 255 * values
    return raw.astype(np.uint8)


def compute_distance_metric(
    row,
    input_5d_stack,
    permuted,
    px_size=0.29,
    crop_size=(64, 160, 96),
):
    """
    Main function to loop over in distributed
    """

    # Dataframe to grab correlations at each res in a df
    df_tmp_corrs = pd.DataFrame()

    # Gets cells for IC dataframe
    if all(elem in list(DatasetFieldsIC.__dict__.values()) for elem in row.index):
        DatasetFields = DatasetFieldsIC

        # get data for cells i and j
        img_i = get_gfp_single_channel_img(
            row[DatasetFields.SourceReadPath1], crop_size=crop_size
        )
        img_j = get_gfp_single_channel_img(
            row[DatasetFields.SourceReadPath2], crop_size=crop_size
        )

        # get the mask for the cell
        mask = get_cell_mask(
            Path(row[DatasetFields.SaveDir])
            / Path(row[DatasetFields.SaveRegPath]).name,
            crop_size=crop_size,
        )
        # kill gfp intensity outside of mask
        img_i[mask == 0] = 0
        img_j[mask == 0] = 0

        mask_i = mask_j = mask

    # Get cells for Morphed cell dataframe
    elif all(
        elem in list(DatasetFieldsMorphed.__dict__.values()) for elem in row.index
    ):
        DatasetFields = DatasetFieldsMorphed

        all_img_i = AICSImage(row[DatasetFields.SourceReadPath1]).data[0, 0, :, :, :, :]
        all_img_j = AICSImage(row[DatasetFields.SourceReadPath2]).data[0, 0, :, :, :, :]

        mask_i = all_img_i[0, :, :, :] + all_img_i[1, :, :, :]
        mask_j = all_img_j[0, :, :, :] + all_img_j[1, :, :, :]
        img_i = all_img_i[2, :, :, :]
        img_j = all_img_j[2, :, :, :]

        # Adjust contast if it is the raw image
        if DatasetFields.SourceReadPath1 == "path_raw_morph_1":
            img_i = pct_normalization_and_8bit(img_i)
            img_j = pct_normalization_and_8bit(img_j)

        img_i = img_i / img_i.max()
        img_j = img_j / img_j.max()

        # kill intensity outside of mask
        img_i[mask_i == 0] = 0
        img_j[mask_j == 0] = 0

    # Get images from 5d stack
    elif any(
        elem in list(DatasetFieldsAverageMorphed.__dict__.values())
        for elem in row.index
    ):
        DatasetFields = DatasetFieldsAverageMorphed

        img_i = input_5d_stack[
            0, row[DatasetFields.StructureIndex1], :, row[DatasetFields.PC_bin], :, :
        ]
        img_j = input_5d_stack[
            0, row[DatasetFields.StructureIndex2], :, row[DatasetFields.PC_bin], :, :
        ]
        # -1 is CELL+NUC mask
        mask = input_5d_stack[0, -1, :, row[DatasetFields.PC_bin], :, :]
        img_i[mask == 0] = 0
        img_j[mask == 0] = 0
        mask_i = mask_j = mask

    try:
        # multi-res comparison
        pyr_corrs = pyramid_correlation(
            img_i, img_j, mask1=mask_i, mask2=mask_j, func=np.mean
        )

        if permuted:
            # comparison when one input is permuted (as baseline correlation)
            pyr_corrs_permuted = pyramid_correlation(
                img_i, img_j, mask1=mask_i, mask2=mask_j, permute=True, func=np.mean
            )

        for k, v in sorted(pyr_corrs.items()):
            if permuted:
                tmp_stat_dict = {
                    "Resolution (micrometers)": px_size * k,
                    "Pearson Correlation": v,
                    "Pearson Correlation permuted": pyr_corrs_permuted[k],
                }
            else:
                tmp_stat_dict = {
                    "Resolution (micrometers)": px_size * k,
                    "Pearson Correlation": v,
                }

            df_tmp_corrs = df_tmp_corrs.append(tmp_stat_dict, ignore_index=True)

        # label stats with StructureName1
        df_tmp_corrs[DatasetFields.StructureName1] = row[DatasetFields.StructureName1]

        # and append row metadata
        df_row_tmp = row.to_frame().T
        df_row_tmp = df_row_tmp.merge(df_tmp_corrs)

        return df_row_tmp

    # Catch ValueError when some of the flattened images have infinity or NaNs
    # This generally throws a ValueError: array must not contain infs or NaNs
    # in the pearson r calculation
    except ValueError:
        return None


def clean_up_results(dist_metric_results, permuted):
    """
    Clean up distributed results.
    """
    df_final = pd.DataFrame()
    for dataframes in dist_metric_results:
        for corr_dataframe in dataframes:
            if corr_dataframe is not None:
                df_final = df_final.append(corr_dataframe)

    # fix up final pairwise dataframe
    df_final = df_final.reset_index(drop=True)

    if permuted:
        df_final["Pearson Correlation gain over random"] = (
            df_final["Pearson Correlation"] - df_final["Pearson Correlation permuted"]
        )

    return df_final


def make_plot(data, plot_dir):
    """
    Seaborn plot of mean corr gain over random vs resolution.
    """
    sns.set(style="ticks", rc={"lines.linewidth": 1.0})

    if "GeneratedStructureName_i" in data.columns:
        fig = plt.figure(figsize=(10, 7))
        ax = sns.pointplot(
            x="Resolution (micrometers)",
            y="Pearson Correlation",
            hue="GeneratedStructureName_i",
            data=data,
            ci=95,
            capsize=0.2,
            palette="Set2",
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.05, 0.95),
            ncol=1,
            frameon=False,
        )
        sns.despine(
            offset=0,
            trim=True,
        )
        # save the plot
        fig.savefig(
            plot_dir / "multi_resolution_image_correlation.png",
            format="png",
            dpi=300,
            transparent=True,
        )
    elif "StructureName_1" in data.columns:

        data = data[data.StructureName_1 != "CELL+NUCLEUS"]
        data = data[data.StructureName_2 != "CELL+NUCLEUS"]

        for this_bin in range(7):
            fig = plt.figure(figsize=(10, 7))
            plot_data = data.loc[(data["bin"] == this_bin)]

            ax = sns.pointplot(
                x="Resolution (micrometers)",
                y="Pearson Correlation",
                hue="StructureName_1",
                data=plot_data,
                ci=95,
                dodge=True,
                capsize=0.2,
                palette="Set2",
            )

            ax.legend(
                loc="best",
                bbox_to_anchor=(1, 0.95),
                ncol=2,
                frameon=False,
            )
            ax.set_title(f"Bin {this_bin}")
            sns.despine(
                offset=0,
                trim=True,
            )

            plt.tight_layout()
            fig.savefig(
                plot_dir / f"multi_resolution_image_correlation_bin_{this_bin}.png",
                format="png",
                dpi=300,
                transparent=True,
            )
