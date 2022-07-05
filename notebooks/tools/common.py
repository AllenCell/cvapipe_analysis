import sys
import yaml
import uuid
import datetime
import concurrent
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from aicscytoparam import cytoparam
from sklearn.decomposition import PCA
from scipy import cluster as spcluster
from skimage import measure as skmeasure
from multiprocessing import shared_memory as smem
from sklearn import discriminant_analysis as sklda
from cvapipe_analysis.tools import io, general, controller, shapespace

def get_map_point_shape(control, device, row, inner_mesh=None, outer_mesh=None):
    device.row = row
    nisos = control.get_number_of_interpolating_points()
    if inner_mesh is None:
        inner_alias = control.get_inner_most_alias_to_parameterize()
        inner_mesh = device.read_map_point_mesh(inner_alias)
    if outer_mesh is None:
        outer_alias = control.get_outer_most_alias_to_parameterize()
        outer_mesh = device.read_map_point_mesh(outer_alias)
    domain, origin = cytoparam.voxelize_meshes([outer_mesh, inner_mesh])
    coords_param, _ = cytoparam.parameterize_image_coordinates(
        seg_mem=(domain>0).astype(np.uint8),
        seg_nuc=(domain>1).astype(np.uint8),
        lmax=control.get_lmax(), nisos=[nisos, nisos]
    )
    domain_nuc = (255*(domain>1)).astype(np.uint8)
    domain_mem = (255*(domain>0)).astype(np.uint8)
    return domain, domain_nuc, domain_mem, coords_param

def get_stack_with_single_cell_from_two_populations(data, scale, bbox, special_raw_contrast={}, text_label="lda"):
    ncells = 15
    # Loading main control for correct gene order and gene contrast settings
    path_config = Path("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/")
    config = general.load_config_file(path_config)
    control = controller.Controller(config)
    # Get figure parameters
    (yyi, yyf, yzi, yzf), bbox, figargs = contact_sheet_params(box_size=bbox, ylow=110, zlow=70)
    views = []
    for ncell, celldata in enumerate(data):
        fig, axs = plt.subplots(1, 3, figsize=(1*scale, 3*scale), **figargs, dpi=150)
        for ch, ax in enumerate(axs):
            ax.axis("off")
            too_big = False
            ch_used = ch+2 if ch < 2 else 3
            proj = Projector(celldata["img"][[0,1,ch_used]], mask_on=True, force_fit=True, box_size=bbox)
            if ch==0: #raw data
                cmap = "gray"
                mode = {"nuc":"max","mem":"max","gfp":"max"}
                vmin, vmax = control.get_optimal_raw_contrast(celldata['gene'])
                if celldata['gene'] in special_raw_contrast:
                    vmin, vmax = special_raw_contrast[celldata['gene']]
                proj.set_vmin_vmax_gfp_values(vmin, vmax)
            if ch==1: #seg data
                cmap = "binary"
                mode = {"nuc":"max","mem":"max","gfp":"mean"}
                vmin, vmax = control.get_optimal_seg_contrast(celldata['gene'])
                proj.set_vmin_vmax_gfp_values(vmin, vmax)
            if ch==2: #seg data
                cmap = "binary"
                mode = {"nuc":"center_nuc","mem":"center_nuc","gfp":"center_nuc"}
                proj.set_gfp_percentiles((20, 100), local=True)
            try:
                proj.set_projection_mode(ax="z", mode=mode)
                proj.compute()
                contourz = proj.get_proj_contours()
                pz = proj.projs["gfp"].copy()
                proj.set_projection_mode(ax="y", mode=mode)
                proj.compute()
                contoury = proj.get_proj_contours()
                py = proj.projs["gfp"].copy()
                im = np.concatenate([py[yyi:yyf, :], pz[yzi:yzf, :]], axis=0)
                ax.imshow(im, cmap=cmap, origin="lower", vmin=proj.gfp_vmin, vmax=proj.gfp_vmax)
                for alias_cont, alias_color in zip(["nuc", "mem"], ["cyan", "magenta"]):
                    [ax.plot(c[:,1], c[:,0]-yyi, lw=0.5, color=alias_color) for c in contoury[alias_cont]]
                    [ax.plot(c[:,1], c[:,0]+(yyf-yyi)-yzi, lw=0.5, color=alias_color) for c in contourz[alias_cont]]
            except:
                too_big = True
                print("Cell is too big")
                pass
            ax.set_ylim(1, im.shape[0])
            ax.set_xlim(1, im.shape[1])
        axs[0].set_title(f"{celldata['gene']}-{celldata['dataset']} - {(ncell+1)%(ncells+1):02d}", fontsize=10)
        axs[1].set_title(f"{text_label}={celldata[text_label]}", fontsize=10)
        axs[2].set_title(f"ID={celldata['CellId']}", fontsize=10)
        plt.tight_layout()
        fig.canvas.draw()
        plt.show(close=True)
        if not too_big:
            views.append(np.array(fig.canvas.renderer._renderer))
    return resize_and_group_images(views)

def normalize_image_to_01_interval(img, vmin, vmax):
    if vmax < 1e-3:
        return img
    img = np.clip(img, vmin, vmax)
    img = (img-vmin)/(vmax-vmin)
    return img

def make_contact_sheet_avgseg_channel(data, bbox, file_prefix=None, contrast=None, ylow=110, zlow=70, nucleus_contour_off=[]):
    # Loading main control for correct gene order and gene contrast settings
    path_config = Path("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/")
    config = general.load_config_file(path_config)
    control = controller.Controller(config)
    genes = control.get_gene_names()
    (yyi, yyf, yzi, yzf), bbox, figargs = contact_sheet_params(box_size=bbox, ylow=ylow, zlow=zlow)
    fig, axs = plt.subplots(len(genes), 1, figsize=(1,len(genes)), **figargs)
    for gene, ax in zip(genes, axs):
        if gene not in data:
            continue
        ax.axis("off")
        instance = data[gene][0]["img"][[0, 1, 2]]
        mode = {"nuc":"center_nuc","mem":"center_nuc","gfp":"center_nuc"}
        proj = Projector(instance, mask_on=True, box_size=bbox, force_fit=True)
        #
        # Z projection
        #
        if contrast is not None:
            vmin, vmaxz = contrast[gene]["z"]
        else:
            vmin, vmax = control.get_optimal_avgseg_contrast(gene)
        proj.set_projection_mode(ax="z", mode=mode)
        proj.compute()
        pz = proj.projs["gfp"].copy()
#         pz = normalize_image_to_01_interval(pz, vmin, vmax)
        contourz = proj.get_proj_contours()
        #
        # Y projection
        #
        if contrast is not None:
            vmin, vmaxy = contrast[gene]["y"]
        else:
            vmin, vmax = control.get_optimal_avgseg_contrast(gene)
        vmax = np.max([vmaxz, vmaxy])
        proj.set_projection_mode(ax="y", mode=mode)
        proj.compute()
        py = proj.projs["gfp"].copy()
#         py = normalize_image_to_01_interval(py, vmin, vmax)
        contoury = proj.get_proj_contours()
        #
        # Combine projections
        #
        im = np.concatenate([py[yyi:yyf, :], pz[yzi:yzf, :]], axis=0)
        im = normalize_image_to_01_interval(im, 0, vmax)
        ax.imshow(im, cmap="inferno", origin="lower", vmin=0, vmax=1)
        for alias_cont, alias_color in zip(["nuc", "mem"], ["cyan", "magenta"]):
            if gene in nucleus_contour_off and alias_cont=="nuc":
                continue
            [ax.plot(c[:,1], c[:,0]-yyi, lw=0.5, color=alias_color) for c in contoury[alias_cont]]
            [ax.plot(c[:,1], c[:,0]+(yyf-yyi)-yzi, lw=0.5, color=alias_color) for c in contourz[alias_cont]]
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(0, im.shape[0])
    if file_prefix is not None:
        plt.savefig(f"{file_prefix}", dpi=300)
    plt.show()

def make_contact_sheet_raw_and_seg_channels(data, bbox, file_prefix, special_raw_contrast={}, nucleus_contour_off=[]):
    # Loading main control for correct gene order and gene contrast settings
    path_config = Path("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/")
    config = general.load_config_file(path_config)
    control = controller.Controller(config)
    genes = control.get_gene_names()
    ncells = len(data[genes[0]])
    # Get figure parameters
    (yyi, yyf, yzi, yzf), bbox, figargs = contact_sheet_params(box_size=bbox, ylow=110, zlow=70)
    # Loop over genes and raw and seg channels
    for ncell in range(ncells):
        for ch, chname in enumerate(["raw", "seg"]):
            fig, axs = plt.subplots(len(genes), 1, figsize=(1,len(genes)), **figargs)
            for gene, ax in zip(genes, axs):
                ax.axis("off")
                instance = data[gene][ncell]["img"][[0, 1, 2+ch]]
                proj = Projector(instance, mask_on=True, box_size=bbox, force_fit=True)
                
                if chname == "seg":
                    mode = {"nuc":"max","mem":"max","gfp":"mean"}
                    vmin, vmax = control.get_optimal_seg_contrast(gene)
                else:
                    mode = {"nuc":"max","mem":"max","gfp":"max"}
                    if gene in ["LMNB1", "NUP153"]:
                        mode["gfp"] = "center_nuc"
                    vmin, vmax = control.get_optimal_raw_contrast(gene)
                    if gene in special_raw_contrast:
                        vmin, vmax = special_raw_contrast[gene]
                proj.set_vmin_vmax_gfp_values(vmin, vmax)
                
                proj.set_projection_mode(ax="z", mode=mode)
                proj.compute()
                pz = proj.projs["gfp"].copy()
                contourz = proj.get_proj_contours()
                proj.set_projection_mode(ax="y", mode=mode)
                proj.compute()
                py = proj.projs["gfp"].copy()
                contoury = proj.get_proj_contours()
                im = np.concatenate([py[yyi:yyf, :], pz[yzi:yzf, :]], axis=0)
                cmap = "binary" if chname=="seg" else "gray"
                imview = ax.imshow(im, cmap=cmap, origin="lower", vmin=proj.gfp_vmin, vmax=proj.gfp_vmax)
                # For raw data we adjust per structure
                if chname == "raw":
                    if gene in ["RAB5A","DSP","CETN2","SLC25A17","PXN"]:#7
                        vmin, vmax = control.get_optimal_raw_contrast(gene)
                    elif gene in ["NPM1", "SMC1A", "AAVS1"]:#8
                        vmin=im[im>0].min()
                        vmax=im[im>0].max()
                    elif gene in ["LMNB1", "NUP153", "GJA1", "TJP1", "CTNNB1", "ACTB", "ACTN1", "MYH10"]:#9
                        vmin = im[im>0].min()
                        vmax = np.percentile(im[im>0], 99)
                    elif gene in ["FBL", "SON", "HIST1H2BJ", "SEC61B", "ATP2A2", "TOMM20", "ST6GAL1", "LAMP1", "TUBA1B"]:#10
                        vmin_abs = im[im>0].min()
                        vmax_abs = im[im>0].max()
                        vrange = vmax_abs - vmin_abs
                        vmin = vmin_abs + 0.05*vrange
                        vmax = vmax_abs - 0.20*vrange
                    imview.set_clim(vmin, vmax)
                for alias_cont, alias_color in zip(["nuc", "mem"], ["cyan", "magenta"]):
                    if gene in nucleus_contour_off and alias_cont=="nuc":
                        continue
                    [ax.plot(c[:,1], c[:,0]-yyi, lw=0.5, color=alias_color) for c in contoury[alias_cont]]
                    [ax.plot(c[:,1], c[:,0]+(yyf-yyi)-yzi, lw=0.5, color=alias_color) for c in contourz[alias_cont]]
                ax.set_xlim(0, im.shape[1])
                ax.set_ylim(0, im.shape[0])
            plt.savefig(f"{file_prefix}_{chname}_{ncell}", dpi=300)
            plt.show()

def contact_sheet_params(box_size, ylow=110, zlow=70):
    bbox = box_size
    # z/y chop factors
    yyi, yzi = ylow, zlow
    yyf, yzf = bbox-yyi, bbox-yzi
    # cell and nuc masks
    yx = (bbox-(yyf-yyi)+bbox-(yzf-yzi))/bbox
    figargs = {"gridspec_kw": {"hspace": 0}, "sharex": True}
    return (yyi, yyf, yzi, yzf), bbox, figargs

def load_single_cell_image(params):
    row = params["row"]
    if params["redirect"]:
        row = redirect_single_cell_path(row)
    producer = io.DataProducer(params["control"])
    producer.set_row(row)
    producer.load_single_cell_data()
    data = producer.data
    if params["alignment"]:
        producer.align_data(force_alignment=True)
        data = producer.data_aligned
    return row.name, data


def load_multiple_single_cell_images_fast(selection, df, control, channels=[3,4,2,7], redirect=False, alignment=True):
    data = {}
    for gene, CellIds in selection.items():
        params = [{
            "row": df.loc[CellId],
            "redirect": redirect,
            "control": control,
            "alignment": alignment
        } for CellId in CellIds]
        with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
            celldata = list(tqdm(executor.map(load_single_cell_image, params), total=len(params), leave=False))
        data[gene] = [{"CellId": idx, "img": img[channels]} for idx, img in celldata]
    return data

def load_multiple_single_cell_images(selection, df, control, channels=[3,4,2,7], redirect=False, check_channels_only=False):
    data = {}
    for gene, CellIds in selection.items():
        producers = []
        for CellId in CellIds:
            row = df.loc[CellId]
            if redirect:
                row = redirect_single_cell_path(row)
            producer = io.DataProducer(control)
            producer.set_row(row)
            producer.load_single_cell_data()
            producer.align_data()
            producers.append((CellId, producer))
            if check_channels_only:
                print(f"Available channels: {[f'{i}-{ch}' for i, ch in enumerate(producer.channels)]}")
                return None
        data[gene] = [{"CellId": i, "img": p.data_aligned[[3, 4, 2, 7]]} for i, p in producers]
    return data

def get_reps_from_lda_walk(pca_lda, mps):
    pcs_back = pca_lda["lda"].walk(mps)#, limit_to_range=lda_values, return_map_points=True)
    pcs_back *= pca_lda["stds"].values
    pcs_back = pd.DataFrame(pcs_back, columns=[f"PC{i}" for i in range(1,1+pca_lda["npcs"])])
    reps_back = pca_lda["pca"].inverse_transform(pcs_back).reshape(len(mps), 65, -1)
    return reps_back

def make_lda_histogram(df, ax, xmin=-3, xmax=3, ymax=2.1, ratio=1.0, verbose=True):
    edges = np.linspace(xmin, xmax, 1+int((xmax-xmin)/0.5))
    for color, (g, df_group) in zip(["black","#FF3264"], df.groupby("Dataset")):
        ax.hist(df_group.LDA, bins=edges, histtype="step", color=color, density=True, lw=2)
        ax.plot([df_group.LDA.mean(), df_group.LDA.mean()], [0, 0.9*ymax], color=color, lw=1, ls="--", zorder=1e10)
#         ax.axvline(x=df_group.LDA.mean(), ymin=0, ymax=0.5*ymax, color=color, lw=1, linestyle="--", zorder=1e10)
        if verbose:
            print(g, df_group.LDA.mean())
    ax.set_xlim(xmin, xmax)

    ax.set_ylim(0, ymax)
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    ax.axes.get_yaxis().set_visible(False)
    sigmas = np.linspace(-2, 2, 5)
    ax.set_xticks(sigmas, [f"{int(s)}$\sigma$" for s in sigmas])
    ax.tick_params(axis="x", which='major', labelsize=8)

    ax2 = ax.secondary_xaxis("top")
    ax2.tick_params(axis="x", direction="in")
    ax2.set_xticks(sigmas, ["" for s in sigmas])

    return

def run_lda_analysis(df_map, managers, return_pca_lda_objs=False):
    
    dsname = [ds for ds in df_map.index.get_level_values("dataset").unique() if ds != "base"][0]
    print(f"Running dataset: {dsname}")
    
    pca_lda = {}
    for gene in df_map.index.get_level_values("structure_name").unique().values:#control.get_gene_names():

        df_gene = df_map.loc[(dsname, gene)]

        CellIds_pt = df_gene.index.values
        CellIds_ct = df_gene.NNCellId.unique()

        rloader_ct = RepsSharedMemoryLoader(managers["control"]["control"])
        rloader_pt = RepsSharedMemoryLoader(managers["perturbed"]["control"])
        reps_ct = rloader_ct.load(CellIds_ct).astype(np.uint8)
        reps_pt = rloader_pt.load(CellIds_pt).astype(np.uint8)
        print(f"mean rep: ct={reps_ct.mean()}, ({len(CellIds_ct)}), pt={reps_pt.mean()}, ({len(CellIds_pt)})")
        '''
        CellIds_ct, reps_ct = get_all_parameterized_intensity_of_seg_channel(CellIds_ct, managers["control"]["device"])
        CellIds_pt, reps_pt = get_all_parameterized_intensity_of_seg_channel(CellIds_pt, managers["perturbed"]["device"])
        print(f"mean rep: ct={reps_ct.mean()}, ({len(CellIds_ct)}), pt={reps_pt.mean()}, ({len(CellIds_pt)})")
        '''
        reps = np.concatenate([reps_ct, reps_pt], axis=0)
        vsize = int(sys.getsizeof(reps)) / float(1 << 20)

        npcs = np.min([32, reps.shape[0]-1])
        pca = PCA(npcs, svd_solver="full")
        pca = pca.fit(reps)
        axes = pca.transform(reps)
        axes = pd.DataFrame(axes, columns=[f"PC{i}" for i in range(1, 1+npcs)])

        groups = np.array([0]*len(CellIds_ct) + [1]*len(CellIds_pt))
        stds = axes.std(axis=0)
        axes /= stds
        axes, pca = sort_pcs(axes, groups, pca)
        axes["Dataset"] = groups
        axes["CellId"] = CellIds_ct.tolist() + CellIds_pt.tolist()
        axes = axes.set_index(["Dataset", "CellId"])

        lda = SimpleBinaryLDA()
        lda = lda.sfit(axes.values, groups)
        lda_values = lda.transform(axes.values).flatten()
        axes["LDA"] = lda_values

        pca_lda[gene] = axes        
        if return_pca_lda_objs:
            pca_lda[gene] = {"axes": axes, "pca": pca, "stds": stds, "npcs": npcs, "lda": lda}

    return pca_lda

def find_cells_nearest_the_mean_of_the_two_populations(pca_lda, ncells=15):
    CellIds = {}
    for gene, axes in pca_lda.items():
        idxs = []
        for ds, df_group in axes.groupby(level="Dataset"):
            dist = np.abs(df_group.LDA-df_group.LDA.mean())
            dist_rank = np.argsort(dist).tolist()
            idxs += df_group.index[dist_rank][:ncells].tolist()
        CellIds[gene] = idxs
    return CellIds

def resize_and_group_images_grays(imgs, offset=0):
    bbox = np.array([im.shape for im in imgs]).max(axis=0)[1:]
    rad = np.max([offset, bbox.max()])
    for i, im in enumerate(imgs):
        shape = im.shape[1:]
        pad = [int(0.5*(rad-s)) for s in shape]
        pad = [(0,0)]+[(p, int(rad-s-p)) for (s, p) in zip(shape, pad)]
        imgs[i] = np.pad(im, pad, mode="constant", constant_values=255)
    imgs = np.array(imgs)
    return imgs

def resize_and_group_images(imgs, offset=0):
    bbox = np.array([im.shape for im in imgs]).max(axis=0)[:-1]
    rad = np.max([offset, bbox.max()])
    for i, im in enumerate(imgs):
        shape = im.shape[:-1]
        pad = [int(0.5*(rad-s)) for s in shape]
        pad = [(p, int(rad-s-p)) for (s, p) in zip(shape, pad)]+[(0,0)]
        imgs[i] = np.pad(im, pad, mode="constant", constant_values=255)
    imgs = np.array(imgs)
    return imgs

def now(text):
    print(text, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def redirect_single_cell_path(row_input):
    row = row_input.copy()
    # redirecting the path to seg images bc some files are corrupted (fixme before resubmission)
    subpath_old = "projects/cvapipe_analysis"
    subpath_new = "datasets/hpisc_single_cell/variance"
    row.crop_seg = row.crop_seg.replace(subpath_old, subpath_new)
    return row

class Masker():
    colors = [
        (255/255., 255/255., 255/255.),
        (100/255., 149/255., 237/255.),
        (128/255., 128/255., 0/255.),
        (240/255., 128/255., 128/255.),
        (200/255., 200/255., 200/255.),
        (200/255., 200/255., 200/255.),
        (200/255., 200/255., 200/255.),
        (200/255., 200/255., 200/255.)
    ]
    patterns = [(1,1,1), (0,1,1), (1,0,1), (1,0,0), (0,1,0), (0,0,1), (1,1,0)]

    def __init__(self, control):
        self.control = control

    def set_data(self, masks, data, colnames=None):
        '''data must be a MxSxF binary matrix, where
        M is the number of masks, S is the number of
        samples and F is the number of features. M
        must match the length of elements of patterns
        colors hardcoded above. Data must have same
        shape but of type float.'''
        self.colnames = colnames
        self.data = self.fix_format(data)
        self.masks = self.fix_format(masks)
        
    def execute(self):
        self.create_groups()
        self.levels = self.create_levels_from_continous_data(self.data)
        self.combine_levels(replace_nan=0)

    def create_groups(self):
        groups = np.ones(self.masks.shape[1:], dtype=int)
        for pid, pattern in enumerate(self.patterns):
            matches = [pattern[c]==value for c, value in enumerate(self.masks)]
            matches = np.logical_and.reduce(matches)
            groups[np.where(matches)] = 2+pid # group = 1 means no hit
        self.groups = groups

    def combine_levels(self, replace_nan=-1):
        levels = self.levels.astype(np.float32)
        for c, mask in enumerate(self.masks):
            levels[c,~mask] = np.nan
        levels = np.nanmin(levels, axis=0)
        if replace_nan > -1:
            levels[np.where(np.isnan(levels))] = replace_nan
            levels = levels.astype(int)
        self.combined_levels = levels

    def get_result_as_dataframes_and_cmap(self, ignore_confidence=True):
        genes = self.control.get_gene_names()
        colnames = genes if self.colnames is None else self.colnames
        dfgrp = pd.DataFrame(self.groups, index=genes, columns=colnames)
        dflvl = pd.DataFrame(self.combined_levels, index=genes, columns=colnames)
        cmap = [(k+1,v) for k, v in enumerate(self.colors)]
        lvls = np.unique(dflvl.values.flatten())
        dflvl.replace(dict([(v,lvls.max()) for v in lvls]), inplace=True)
        return dfgrp, dflvl, dict(cmap)

    @staticmethod
    def fix_format(x):
        if x.ndim==2:
            return x.reshape(1,*x.shape)
        return x
        
    @staticmethod
    def create_levels_from_continous_data(x, bins=[-np.inf,0.05,0.25,np.inf], labels=[7, 4, 1]):
        levels = pd.cut(x.flatten(), bins=bins, labels=labels).tolist()
        return np.array(levels).reshape(*x.shape)

    @staticmethod
    def reduce_to_diag(xs, swap_axs=False):
        d = np.array([np.diag(x) for x in xs]).T
        d = d.reshape(1,*d.shape)
        if swap_axs:
            d = np.swapaxes(d, 0, -1)
        return d
    
class CorrTable:
    
    def __init__(self, control):
        
        self.cols = []
        self.precision = 3
        self.control = control
        self.phenos = ["ct", "pt"]
        self.aliases = ["M1M2", "M3"]
        self.data_folder = f"{control.get_staging()}/concordance/plots/"
        self.filters = {"Corr": ("gt",0.03), "Delta": ("gt",0.02), "Swap": ("lt",0.05), "ABSpct": ("gt", 0.2), "Pct": ("gt", 0.2)}
        path_cvapipe = Path(control.get_staging()).parent
        self.datasets = {
            "M1M2": {
                "ct": f"{path_cvapipe}/local_staging_variance_m1m2",
                "pt": f"{path_cvapipe}/local_staging_m1m2"
            },
            "M3": {
                "ct": f"{path_cvapipe}/local_staging_variance_m3",
                "pt": f"{path_cvapipe}/local_staging_m3"
            }
        }
        self.comparisons = {
            "Control vs. M1M2": [("M1M2", "ct"), ("M1M2", "pt")],
            "M1M2 vs. M3": [("M1M2", "pt"), ("M3", "pt")],
            "Control vs. M3": [("M3", "ct"), ("M3", "pt")]
        }

        self.load_correlations()
        self.load_swaps()
        self.make_dataframe()
        self.calculate_deltas()
#         self.calculate_pct_change()

    def get_unique_combinations_of_comparisons(self):
        df = pd.DataFrame([])
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            for a, p in zip([a1, a2], [p1, p2]):
                df = df.append({"alias": a, "phenotype": p}, ignore_index=True)
        return df.drop_duplicates().reset_index(drop=True)

    def load_correlations(self):
        for _, comb in self.get_unique_combinations_of_comparisons().iterrows():
            path = Path(self.datasets[comb.alias][comb.phenotype]) / "concordance"
            control, dev = get_managers_from_step_path(path)
            variables = control.get_variables_values_for_aggregation()
            df_agg = shapespace.ShapeSpaceBasic.get_aggregated_df(variables)
            df_agg = df_agg.drop(columns=["structure"]).drop_duplicates().reset_index(drop=True)
            df, _ = dev.get_mean_correlation_matrix_of_reps(df_agg.loc[0])
            col = self.df_to_col(df, col_name=f"Corr_{comb.alias}_{comb.phenotype}")
            self.cols.append(col)
                
    def load_swaps(self):
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            sufix = f"{a1}_{p1}_{a2}_{p2}"
            fname = f"../SwapCalculation/swap_{sufix}.csv"
            try:
                df = pd.read_csv(fname, index_col=0)
            except:
                print(f"WARNING: No swap found for {sufix}")
                return
            name = f"Swap_{sufix}"
            col = self.df_to_col(df, col_name=name)
            self.cols.append(col)
                        
    def make_dataframe(self):
        df = pd.DataFrame(self.cols).T
        df = df.dropna(axis=1, how="all")
        self.df = df
    
    def calculate_deltas(self):
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            name1 = f"Corr_{a1}_{p1}"
            name2 = f"Corr_{a2}_{p2}"
            name = f"Delta_{a1}_{p1}_{a2}_{p2}"
            self.df[name] = self.df[name1]-self.df[name2]

    def calculate_pct_change(self):
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            name1 = f"Corr_{a1}_{p1}"
            name2 = f"Corr_{a2}_{p2}"
            name = f"Pct_{a1}_{p1}_{a2}_{p2}"
            self.df[name] = (self.df[name1]-self.df[name2])/self.df[name1]
            name = f"ABSpct_{a1}_{p1}_{a2}_{p2}"
            self.df[name] = (self.df[name1]-self.df[name2]).abs()/self.df[name1]

    def round_values(self):
        self.df = self.df.round(decimals=self.precision)
            
    def apply_filters(self, metrics, cast=True, replace_with_nan=False):
        if cast:
            self.round_values()
        for col in self.df.columns:
            for name, (comp, thresh) in self.filters.items():
                if name in col:
                    values = self.df[col].abs()
                    values = values > thresh if comp=="gt" else values < thresh
                    self.df[f"{col}_valid"] = values
        self.df = self.df[sorted(self.df.columns)]
        
        for cid, (comp_name, ((a1, p1), (a2, p2))) in enumerate(self.comparisons.items()):
            checks = {}
            checks["Corr"] = self.df[f"Corr_{a1}_{p1}_valid"]
            checks["Delta"] = self.df[f"Delta_{a1}_{p1}_{a2}_{p2}_valid"]
            checks["Swap"] = self.df[f"Swap_{a1}_{p1}_{a2}_{p2}_valid"]
#             checks["Pct"] = self.df[f"Pct_{a1}_{p1}_{a2}_{p2}_valid"]
#             checks["ABSpct"] = self.df[f"ABSpct_{a1}_{p1}_{a2}_{p2}_valid"]
            checks = [v.values for k, v in checks.items() if k in metrics]
            checks = np.prod(checks, axis=0).astype(bool)
            self.df[comp_name] = checks
            if replace_with_nan:
                if "Delta" in metrics:
                    self.df.loc[checks==False, [f"Delta_{a1}_{p1}_{a2}_{p2}"]] = np.nan
#                 if "Pct" in metrics:
#                     self.df.loc[checks==False, [f"Pct_{a1}_{p1}_{a2}_{p2}"]] = np.nan
#                 if "ABSpct" in metrics:
#                     self.df.loc[checks==False, [f"ABSpct_{a1}_{p1}_{a2}_{p2}"]] = np.nan

    def cluster(self, cols, thresh=0.2, show=False):
        X = self.df[cols].values
        Z = spcluster.hierarchy.linkage(X, "average")
        if show:
            fig, ax = plt.subplots(1, 1, figsize=(32, 16))
            _ = spcluster.hierarchy.dendrogram(
                Z, labels=self.get_indices_as_strings(self.df), leaf_rotation=90
            )
            ax.set_ylim(-0.1, 1.5)
            plt.tight_layout()
            plt.show()
        self.df["Cluster"] = spcluster.hierarchy.fcluster(Z, t=thresh, criterion="distance")

    def get_columns_to_display(self):
        cols = ["Cluster"] + [k for k in self.comparisons.keys()]
        for p in self.phenos:
            for a in self.aliases:
                cols.append(f"Corr_{a}_{p}")
        for s in ["Delta", "Swap"]:#, "Pct", "ABSpct"]:
            for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
                cols.append(f"{s}_{a1}_{p1}_{a2}_{p2}")
        return cols
        
    def get_df(self, remove_symm=False, group=None, return_col_names=False):
        cols = self.get_columns_to_display()
        df = self.df.copy()
        if group is not None:
            df["flag"] = True
            for cid, cval in enumerate(group):
                df.flag = df.flag * ((df[f"Comp{cid}"]==cval))
            df = df.loc[df.flag==True]
        df = df[cols]
        if remove_symm:
            df = self.remove_rows_with_symmetric_indices(df)
            
        SCols = ["S1", "S2"]
        df.index.set_names(SCols, inplace=True)
        df = df.reset_index()
        for col in SCols:            
            df[col] = pd.Categorical(df[col].values, categories=self.control.get_gene_names(), ordered=True)
        df = df.sort_values(by=["Cluster", "S1", "S2"])
        cols = df.columns.tolist()
        df = df[["Cluster"]+[col for col in cols if col != "Cluster"]]
        if return_col_names:
            bcols = [f for f in self.comparisons.keys()]
            ccols = [f for f in df.columns if "Corr" in f]
            dcols = [f for f in df.columns if "Delta" in f]
            scols = [f for f in df.columns if "Swap" in f]
        return df, (bcols, ccols, dcols, scols)
    
    @staticmethod
    def get_indices_as_strings(df):
        return [f"{s1}-{s2}" for (s1, s2) in df.index]
    
    @staticmethod
    def remove_rows_with_symmetric_indices(df):
        df["flag"] = False
        for s1, s2 in df.index:
            if (s1==s2) | df.at[(s1, s2), "flag"]:
                continue
            if (s2, s1) in df.index:
                df.at[(s2, s1), "flag"] = True
        df = df.loc[df.flag==False].drop(columns=["flag"])
        return df
    
    @staticmethod
    def read_df(path, icol=0):
        try:
            df = pd.read_csv(path, index_col=icol)
        except:
            return pd.DataFrame([])
        
        if "self" in df.columns:
            df = df.drop(columns=["self"])
        return df
    
    @staticmethod
    def df_to_col(df, col_name):
        col = df.unstack()
        col.name = col_name
        return col
    
    @staticmethod
    def convert_swap_into_ci_levels(swapmx, labels=[7, 4, 1]):
        levels = []
        for c in range(swapmx.shape[1]):
            x = pd.cut(swapmx[:, c], bins=[-np.inf,0.05,0.25,np.inf], labels=labels).tolist()
            levels.append(x)
        levels = np.array(levels).T
        return levels
    
def save_html_table(df, columns, table, filename):
    comp_cols, corr_cols, delta_cols = columns[:3]
    html = df.style\
    .background_gradient("tab20", axis=0, subset=["Cluster"])\
    .applymap(lambda v: f"background-color: {table.control.get_gene_color(v)}", subset=["S1"])\
    .background_gradient("Greys", vmin=0, vmax=1, axis=0, subset=comp_cols)\
    .background_gradient("PRGn", vmin=-0.1, vmax=0.1, axis=1, subset=delta_cols)\
    .background_gradient("seismic", vmin=-0.2, vmax=0.2, axis=1, subset=corr_cols)\
    .applymap(lambda v: 'opacity: 20%;' if (np.abs(v)<=table.filters["Delta"][-1]) else None, subset=delta_cols)\
    .applymap(lambda v: 'opacity: 20%;' if (np.abs(v)<=table.filters["Corr"][-1]) else None, subset=corr_cols)\
    .format(na_rep=".", precision=3).highlight_null('white')\
    .set_table_styles([dict(selector="th",props=[('max-width', '30px')]), dict(selector="th.col_heading", props=[('transform', 'rotateZ(-90deg)'), ('height', '180px')])])\
    .to_html()
    with open(filename,'w') as f:
        f.write(html)
#     .background_gradient("PRGn", vmin=0, vmax=2, axis=1, subset=pct_cols)\
#     .background_gradient("PRGn", vmin=-2, vmax=2, axis=1, subset=abspct_cols)\
#   .applymap(lambda v: 'opacity: 20%;' if (np.abs(v)<=table.filters["Pct"][-1]) else None, subset=abspct_cols)\
#     .applymap(lambda v: 'opacity: 20%;' if (np.abs(v)<=table.filters["ABSpct"][-1]) else None, subset=pct_cols)\
    
class SwapCalculator():
    
    def __init__(self, config):
        self.setup_comparisons(config)
        self.load_agg_correlation_data()
        
    def setup_comparisons(self, config):
        '''Paths to the matched datasets. ct and pt refer to control
        and perturbed datasets. Below we specify the combinations
        of tests we want to perform. The combinations are specified
        using a name, then a pair of tuples. Each tuple must include
        a matched-dataset name and condition (ct or pt).'''
        path_cvapipe = Path(config["project"]["local_staging"]).parent
        self.datasets = {
            "M1M2": {
                "ct": f"{path_cvapipe}/local_staging_variance_m1m2",
                "pt": f"{path_cvapipe}/local_staging_m1m2"
            },
            "M3": {
                "ct": f"{path_cvapipe}/local_staging_variance_m3",
                "pt": f"{path_cvapipe}/local_staging_m3"
            }}

        self.comparisons = {
            "ControlM1M2 vs. M1M2": [("M1M2", "ct"), ("M1M2", "pt")],
            "M1M2 vs. M3": [("M1M2", "pt"), ("M3", "pt")],
            "ControlM3 vs. M3": [("M3", "ct"), ("M3", "pt")],
            "ControlM1M2 vs. ControlM3": [("M1M2", "ct"), ("M3", "ct")]
        }

    def get_unique_combinations_of_comparisons(self):
        df = pd.DataFrame([])
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            for a, p in zip([a1, a2], [p1, p2]):
                df = df.append({"alias": a, "phenotype": p}, ignore_index=True)
        return df.drop_duplicates().reset_index(drop=True)
    
    def load_agg_correlation_data(self):
        '''Loads the correlation matrix for each pair (alias, condition)
        as well as the number of cells used to calculate the correlations.'''
        data = []
        for _, comb in self.get_unique_combinations_of_comparisons().iterrows():
            path = Path(self.datasets[comb.alias][comb.phenotype]) / "concordance"
            control, dev = get_managers_from_step_path(path)
            variables = control.get_variables_values_for_aggregation()
            df_agg = shapespace.ShapeSpaceBasic.get_aggregated_df(variables)
            df_agg = df_agg.drop(columns=["structure"]).drop_duplicates().reset_index(drop=True)
            for _, row in df_agg.iterrows():
                df_corr, prefix, df_size = dev.get_mean_correlation_matrix_of_reps(row, return_ncells=True)
                dfs = {"corr": df_corr, "size": df_size}
                data.append((comb.alias, comb.phenotype, prefix, dfs))
        self.data = dict()
        for k1, k2, k3, value in data:
            self.data.setdefault(k1, {}).setdefault(k2, {}).update({k3: value})

    def set_df_of_error_curves(self, df, colname):
        self.colname = colname
        self.df_error = df.copy()        

    def calculate_swaps(self, a1, p1, a2, p2):
        '''By defining the prefix as below we are explicitly using the fact
        that matched-datasets have been processed using a single shape mode
        binned in a single bin.'''
        np.random.seed(42)
        prefix = [k for k in self.data[a1][p1].keys()][0]
        genes = self.data[a1][p1][prefix]["corr"].index
        df_swap = pd.DataFrame(0, index=genes, columns=genes)
        for gid1, gene1 in tqdm(enumerate(genes), total=len(genes), leave=False):
            for gid2, gene2 in enumerate(genes):
                if gid2 >= gid1:
                    n11 = self.data[a1][p1][prefix]["size"].at[gene1, "ncells"]
                    n12 = self.data[a1][p1][prefix]["size"].at[gene2, "ncells"]
                    n21 = self.data[a2][p2][prefix]["size"].at[gene1, "ncells"]
                    n22 = self.data[a2][p2][prefix]["size"].at[gene2, "ncells"]
                    n1 = np.min([n11, n12])#int(0.5*(n11+n12))
                    n2 = np.min([n21, n22])#int(0.5*(n21+n22))
                    corr1 = self.data[a1][p1][prefix]["corr"].at[gene1, gene2]
                    corr2 = self.data[a2][p2][prefix]["corr"].at[gene1, gene2]
                    ratio1 = self.get_ratio(gene1, gene2, n1)
                    ratio2 = self.get_ratio(gene1, gene2, n2)
                    scale1 = np.abs(corr1)*(ratio1-1)
                    scale2 = np.abs(corr2)*(ratio2-1)
                    swap = self.bootstrap(corr1, scale1, corr2, scale2)
                    df_swap.loc[gene1, gene2] = df_swap.loc[gene2, gene1] = swap
        return df_swap

    def get_ratio(self, gene1, gene2, n):
        '''Remove 1 from the ratio to get a proxy for the variation
        around the true value. Limiting scale to 0 in case it goes
        negative due to stochastic fluctuations.'''
        n = 400 if n > 400 else n
        index = (gene1, gene2, n)
        if index in self.df_error.index:
            return np.max([1, self.df_error.at[index, self.colname]])
        return np.max([1, self.df_error.at[(gene2, gene1, n), self.colname]])

    @staticmethod
    def bootstrap(x1, scale1, x2, scale2):
        NRUNS = 10000
        x1_new = np.random.normal(loc=x1, scale=scale1, size=NRUNS)
        x2_new = np.random.normal(loc=x2, scale=scale2, size=NRUNS)
        count = np.logical_xor(x1>=x2, x1_new>=x2_new)
        return count.sum()/NRUNS

    @staticmethod
    def expdist(m, s):
        x = np.linspace(m-4*s, m+4*s, 64)
        y = np.exp(-0.5*((x-m)/s)**2)
        y /= y.sum()
        return x, y

    def visualize_distributions_for(self, gene1, gene2):
        nc = len(self.comparisons)
        fig, axs = plt.subplots(1,nc, figsize=(3*nc,1.5))
        xmin, xmax = [], []
        for ax, (_, ((a1, p1), (a2, p2))) in zip(axs, self.comparisons.items()):
            prefix = [k for k in self.data[a1][p1].keys()][0]
            m1 = self.data[a1][p1][prefix]["corr"].loc[gene1, gene2]
            n11 = self.data[a1][p1][prefix]["size"].at[gene1, "ncells"]
            n12 = self.data[a1][p1][prefix]["size"].at[gene2, "ncells"]
            m2 = self.data[a2][p2][prefix]["corr"].loc[gene1, gene2]
            n21 = self.data[a2][p2][prefix]["size"].at[gene1, "ncells"]
            n22 = self.data[a2][p2][prefix]["size"].at[gene2, "ncells"]
            n1 = np.min([n11, n12])
            n2 = np.min([n21, n22])
            r1 = self.get_ratio(gene1, gene2, n1)
            r2 = self.get_ratio(gene1, gene2, n2)
            x1, y1 = SwapCalculator.expdist(m1, m1*(r1-1))
            ax.plot(x1, y1)
            x2, y2 = SwapCalculator.expdist(m2, m2*(r2-1))
            ax.plot(x2, y2)
            xmin.append(np.min([x1, x2]))
            xmax.append(np.max([x1, x2]))
            ax.set_title(f"{a1}_{p1} vs. {a2}_{p2}\n{m1:.3f} ({n1}), {m2:.3f} ({n2})")
        for axi, ax in enumerate(axs):
            if not axi:
                ax.set_ylabel(f"{gene1}-{gene2}", fontsize=14)
            ax.set_xlim(np.min(xmin), np.max(xmax))
        plt.show()
    
    def execute(self):
        dfs = {}
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            df_swap = self.calculate_swaps(a1, p1, a2, p2)
            dfs[f"{a1}_{p1}_{a2}_{p2}"] = df_swap
        return dfs
    
class RepsSharedMemoryReader(io.LocalStagingIO):
    
    rep_length = 532610
    
    def __init__(self, control, shm_id):
        self.shm_id = shm_id
        super().__init__(control)

    def read_representation_as_boolean(self, eindex):
        i, index = eindex
        rep = self.read_parameterized_intensity_of_alias(index, "STR").astype(bool).flatten()
        if rep is not None:
            ptr_reps = smem.SharedMemory(name=self.shm_id)
            shm_reps = np.ndarray((self.ncells, self.rep_length), dtype=bool, buffer=ptr_reps.buf)
            shm_reps[i] = rep
            #ptr_reps.close()
            #ptr_reps.unlink()
        return

    def load(self, CellIds):
        self.CellIds = CellIds
        self.ncells = len(CellIds)
        func = self.read_representation_as_boolean
        with tqdm(total=self.ncells, leave=False) as pbar:
            with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
                futures = {executor.submit(func, eid): eid for eid in enumerate(self.CellIds)}
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    
class RepsSharedMemoryLoader():
    def __init__(self, control):
        self.shm_id = uuid.uuid4().hex[:8]
        self.loader = RepsSharedMemoryReader(control, self.shm_id)
        
    def load(self, CellIds):
        ncells = len(CellIds)
        reps = np.zeros((ncells, RepsSharedMemoryReader.rep_length), dtype=bool)
        ptr_reps = smem.SharedMemory(create=True, size=reps.nbytes, name=self.shm_id)
        shm_reps = np.ndarray(reps.shape, dtype=reps.dtype, buffer=ptr_reps.buf)
        shm_reps[:] = reps[:]
        #ptr_reps.close()
        #ptr_reps.unlink()
        self.loader.load(CellIds)
        reps[:] = shm_reps[:]
        return reps

def get_all_parameterized_intensity_of_seg_channel(CellIds, device):
#     raise ValueError("Function deprecated. Please use RepsSharedMemoryLoader")
    ncores = multiprocessing.cpu_count()
    idsch = zip(*[(i,"STR") for i in CellIds])
    with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
        reps = list(tqdm(
            executor.map(device.read_parameterized_intensity_of_alias, *idsch), leave=False, total=len(CellIds)
        ))
    reps = [(i, r.flatten()) for (i, r) in zip(CellIds, reps) if r is not None]
    CellIds = np.array([i for (i, r) in reps])
    reps = np.array([r for (i, r) in reps], dtype=bool)
    return CellIds, reps

def sort_pcs(axes, groups, pca=None):
    # Control (group=0) should be represented by negative.
    # values. Remember data is centered.
    for pcid, pc in enumerate(axes.columns):
        if axes[pc].values[groups==0].mean() > 0:
            axes[pc] *= -1
            if pca is not None:
                pca.components_[pcid] *= -1
    return axes, pca

def get_managers_from_step_path(path):
    with open(Path(path)/"parameters.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    control = controller.Controller(config)
    device = io.LocalStagingIO(control)
    return control, device

def setup_cvapipe_for_matched_dataset(config, dataset, step_to_use="preprocessing"):
    dsmanagers = {}
    for pheno, path in dataset.items():
        step_path = Path(path) / step_to_use
        control, device = get_managers_from_step_path(step_path)
#         config["project"]["local_staging"] = path
#         control = controller.Controller(config)
#         device = io.LocalStagingIO(control)
        dsmanagers[pheno] = {"control": control, "device": device}
    return dsmanagers

def get_all_norm_parameterized_intensity_of_seg_channel(CellIds, device):
    ncores = multiprocessing.cpu_count()
    idsch = zip(*[(i,"STR") for i in CellIds])
    with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
        reps = list(tqdm(
            executor.map(device.read_normalized_parameterized_intensity_of_alias, *idsch), leave=False, total=len(CellIds)
        ))
    reps = [r for r in reps if r is not None]
    reps = np.array(reps)
    return reps

class SimpleBinaryLDA(sklda.LinearDiscriminantAnalysis):
    def __init__(self):
        super().__init__(solver="eigen", store_covariance=True, shrinkage="auto")
    def sfit(self, X, y):
        self.data = (X, y)
        self.fit(X, y)
        self.axis = np.matmul(np.linalg.inv(self.covariance_), np.diff(self.means_, axis=0).T).flatten()
        self.centroid = X.mean(axis=0).flatten()
        self.versor = self.axis / np.linalg.norm(self.axis)
        self.fix_versor_orientation()
        return self
    def fix_versor_orientation(self):
        dist = []
        X, y = self.data
        for g in [0, 1]:
            xm = X[y==g, :].mean(axis=0)
            dist.append(np.linalg.norm(xm-(self.centroid+self.versor)))
        if np.argmin(dist) == 0:
            self.versor *= -1
    def transform(self, X):
        lda_values = np.zeros(X.shape[0], dtype=np.float32)
        for i, x in enumerate(X):
            lda_values[i] = ((x-self.centroid)*self.versor).sum()
        return lda_values
    def display2d(self):
        X, y = self.data
        _, ax = plt.subplots(1,1)
        ax.set_aspect("equal")
        for c in np.unique(y):
            ax.scatter(X[y==c,0], X[y==c,1])
            xo = self.centroid[0]
            xf = self.axis[0]
            yo = self.centroid[1]
            yf = self.axis[1]
        ax.plot([xo, xo+xf], [yo, yo+yf], 'k')
        ax.plot([xo, xo-xf], [yo, yo-yf], 'k')
        ax.scatter(xo,yo,s=50,c='k')
        ax.plot([xo,xo+self.versor[0]], [yo,yo+self.versor[1]], lw=5, color='k')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        plt.show()
    def walk(self, map_points, limit_to_range=None, return_map_points=False):
        map_points = np.array(map_points)
        if limit_to_range is not None:
            map_points = (map_points-map_points.min())/map_points.ptp()
            vmin = limit_to_range.min()
            vmax = limit_to_range.max()
            map_points = vmin + map_points*(vmax-vmin)
        coords = np.matmul(map_points.reshape(-1,1), self.versor.reshape(1,-1))
        if return_map_points:
            return coords, map_points
        return coords

class Projector():
    # Expects a 3 channels 3D image with following channels:
    # 0 - nucleus (binary mask)
    # 1 - membrane (binary mask)
    # 2 - structure (np.float32)
    verbose = False
    PANEL_SIZE = 2 #Size of a matplotlib panel
    gfp_pcts = [10, 90]
    CMAPS = {"nuc": "gray", "mem": "gray", "gfp": "inferno"}

    def __init__(self, data, force_fit=False, box_size=300, mask_on=False):
        self.data = {
            "nuc": data[0].copy().astype(bool),
            "mem": data[1].copy().astype(bool),
            "gfp": data[2].copy().astype(np.float32)
        }
        self.local_pct = False
        self.box_size = box_size
        self.force_fit = force_fit
        self.bbox = self.get_bbox_of_chanel("mem")
        self.tight_crop()
        self.pad_data()
        self.gfp_vmin = None
        self.gfp_vmax = None
        if mask_on:
            self.mask_gfp_channel()

    def mask_gfp_channel(self):
        mem = self.data["mem"]
        self.data["gfp"][mem==0] = 0
        
    def set_verbose_on(self):
        self.verbose = True
        
    def set_projection_mode(self, ax, mode):
        # mode is a dict with keys: nuc, mem and gfp
        self.proj_ax = ax
        self.proj_mode = mode

    def view(self, alias, ax, chopy=None):
        proj = self.projs[alias]
        args = {}
        if chopy is not None:
            proj = proj[chopy:-chopy]
        if alias == "gfp":
            args = {"vmin": self.gfp_vmin, "vmax": self.gfp_vmax}
        return ax.imshow(proj, cmap=self.CMAPS[alias], origin="lower", **args)
        
    def display(self, save):
        for alias, proj in self.projs.items():
            fig, ax = plt.subplots(1, 1, figsize=(self.PANEL_SIZE, self.PANEL_SIZE))
            _ = self.view(alias=alias, ax=ax)
            ax.axis("off")
            if save is not None:
                fig.savefig(f"{save}_{alias}.png", dpi=150)
            plt.show()

    def get_projections(self):
        self.projs = {}
        ax = ["z", "y", "x"].index(self.proj_ax)
        for alias, img in self.data.items():
            if self.proj_mode[alias] == "max":
                p = img.max(axis=ax)
            if self.proj_mode[alias] == "mean":
                p = img.mean(axis=ax)
            if self.proj_mode[alias] == "top_nuc":
                zc, yc, xc = [int(np.max(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc]
                if self.proj_ax == "y":
                    p = img[:, yc]
                if self.proj_ax == "x":
                    p = img[:, :, xc]
            if self.proj_mode[alias] == "center_nuc":
                zc, yc, xc = [int(np.mean(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc]
                if self.proj_ax == "y":
                    p = img[:, yc]
                if self.proj_ax == "x":
                    p = img[:, :, xc]
            if self.proj_mode[alias] == "center_mem":
                zc, yc, xc = [int(np.mean(u)) for u in np.where(self.data["mem"])]
                if self.proj_ax == "z":
                    p = img[zc]
                if self.proj_ax == "y":
                    p = img[:, yc]
                if self.proj_ax == "x":
                    p = img[:, :, xc]
            if self.proj_mode[alias] == "max_buffer_center_nuc":
                buf = 3
                zc, yc, xc = [int(np.mean(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc-buf:zc+buf].max(axis=ax)
                if self.proj_ax == "y":
                    p = img[:, yc-buf:yc+buf].max(axis=ax)
                if self.proj_ax == "x":
                    p = img[:, :, xc-buf:xc+buf].max(axis=ax)
            if self.proj_mode[alias] == "max_buffer_top_nuc":
                buf = 3
                zc, yc, xc = [int(np.max(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc-buf:zc+buf].max(axis=ax)
                if self.proj_ax == "y":
                    p = img[:, yc-buf:yc+buf].max(axis=ax)
                if self.proj_ax == "x":
                    p = img[:, :, xc-buf:xc+buf].max(axis=ax)
            if self.verbose:
                print(f"Image shape: {img.shape}, slices used: ({zc},{yc},{xc})")
            self.projs[alias] = p

    def set_gfp_percentiles(self, pcts, local=False):
        self.gfp_pcts = pcts
        self.local_pct = local

    def set_vmin_vmax_gfp_values(self, vmin, vmax):
        self.gfp_vmin = vmin
        self.gfp_vmax = vmax
        
    def set_gfp_colormap(self, cmap):
        self.CMAPS["gfp"] = cmap
            
    def calculate_gfp_percentiles(self):
        if self.gfp_vmin is not None:
            if self.verbose:
                print("vmin/vmax values already set...")
            return
        data = self.data
        if self.local_pct:
            data = self.projs
        data = data["gfp"][data["mem"]>0]
        self.gfp_vmin = np.percentile(data, self.gfp_pcts[0])
        self.gfp_vmax = np.percentile(data, self.gfp_pcts[1])
        if self.verbose:
            print(f"GFP min/max: {self.gfp_vmin:.3f} / {self.gfp_vmax:.3f}")
        
    def compute(self, scale_bar=None):
        self.get_projections()
        self.calculate_gfp_percentiles()
        if scale_bar is not None:
            self.stamp_scale_bar(**scale_bar)

    def project(self, save=None, scale_bar=None):
        self.compute(scale_bar=scale_bar)
        self.display(save)

    def project_on(self, alias, ax, chopy=None, scale_bar=None):
        self.compute(scale_bar=scale_bar)
        return self.view(alias=alias, ax=ax, chopy=chopy)
                
    def get_bbox_of_chanel(self, channel):
        img = self.data[channel]
        z, y, x = np.where(img)
        sz, sy, sx = img.shape
        zmin, zmax = np.max([0, np.min(z)]), np.min([sz-1, np.max(z)])
        ymin, ymax = np.max([0, np.min(y)]), np.min([sy-1, np.max(y)])
        xmin, xmax = np.max([0, np.min(x)]), np.min([sx-1, np.max(x)])
        return xmin, xmax, ymin, ymax, zmin, zmax

    def tight_crop(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bbox
        for alias, img in self.data.items():
            img = img[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
            if self.force_fit:
                zf = np.min([ymax-ymin, self.box_size])
                yf = np.min([ymax-ymin, self.box_size])
                xf = np.min([xmax-xmin, self.box_size])
                img = img[:zf, :yf, :xf]
            self.data[alias] = img

    def pad_data(self):
        mingfp = self.data["gfp"][self.data["mem"]>0].min()
        for alias, img in self.data.items():
            shape = img.shape
            pad = [int(0.5*(self.box_size-s)) for s in shape]
            pad = [(p, int(self.box_size-s-p)) for (s, p) in zip(shape, pad)]
            if np.min([np.min([i,j]) for i, j in pad]) < 0:
                raise ValueError(f"Box of size {self.box_size} invalid for image of size: {shape}.")
            self.data[alias] = np.pad(img, pad, mode="constant", constant_values=0 if alias != "gfp" else mingfp)

    def stamp_scale_bar(self, pixel_size=0.108, length=5):
        xc = int(0.5*self.box_size)
        n = int(length/pixel_size)
        self.projs["nuc"][20:30, xc:xc+n] = True

    def get_proj_contours(self):
        cts = {}
        for alias in ["nuc", "mem"]:
            im = self.projs[alias]
            cts[alias] = skmeasure.find_contours(im, 0.5)
        return cts

    @staticmethod
    def get_shared_morphed_max_based_on_pct_for_zy_views(instances, pct, mode, func=np.max, include_vmin_as_zero=True, nonzeros_only=True):
        vmax = {"z":[], "y":[]}
        for img in instances:
            for ax in ["z", "y"]:
                proj = Projector(img, force_fit=True)
                proj.set_projection_mode(ax=ax, mode=mode)
                proj.compute()
                values = proj.projs["gfp"].flatten()
                if nonzeros_only:
                    values = values[values>0.0]
                v = 0
                if len(values) > 0:
                    v = np.percentile(values, pct)
                vmax[ax].append(v)
        if include_vmin_as_zero:
            return dict([(ax, (0,func(vals))) for ax, vals in vmax.items()])
        return dict([(ax, func(vals)) for ax, vals in vmax.items()])

    @staticmethod
    def get_shared_gfp_range_for_zy_views_old(instances, pcts, mode):
        minmax = {"z": [], "y": []}
        for k, cellinfos in instances.items():
            for cellinfo in cellinfos:
                img = cellinfo["img"]
                for ax in ["z", "y"]:
                    proj = Projector(img, force_fit=True)
                    proj.set_projection_mode(ax=ax, mode=mode)
                    proj.compute()
                    values = proj.projs["gfp"].flatten()#[proj.projs["gfp"]>0]
                    if len(values):
                        minmax[ax].append(np.percentile(values, pcts))
        print(minmax)
        return dict([(ax, (np.min(vals), np.max(vals))) for ax, vals in minmax.items()])
    
class SwapCalculatorOld():
    
    def __init__(self, config):
        self.setup_comparisons(config)
        self.load_agg_correlation_data()
        
    def setup_comparisons(self, config):
        '''Paths to the matched datasets. ct and pt refer to control
        and perturbed datasets. Below we specify the combinations
        of tests we want to perform. The combinations are specified
        using a name, then a pair of tuples. Each tuple must include
        a matched-dataset name and condition (ct or pt).'''
        path_cvapipe = Path(config["project"]["local_staging"]).parent
        self.datasets = {
            "M1M2": {
                "ct": f"{path_cvapipe}/local_staging_variance_m1m2",
                "pt": f"{path_cvapipe}/local_staging_m1m2"
            },
            "M3": {
                "ct": f"{path_cvapipe}/local_staging_variance_m3",
                "pt": f"{path_cvapipe}/local_staging_m3"
            }}

        self.comparisons = {
            "Control vs. M1M2": [("M1M2", "ct"), ("M1M2", "pt")],
            "M1M2 vs. M3": [("M1M2", "pt"), ("M3", "pt")],
            "Control vs. M3": [("M3", "ct"), ("M3", "pt")],
            "ControlM1M2 vs. ControlM3": [("M1M2", "ct"), ("M3", "ct")]
        }

    def get_unique_combinations_of_comparisons(self):
        df = pd.DataFrame([])
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            for a, p in zip([a1, a2], [p1, p2]):
                df = df.append({"alias": a, "phenotype": p}, ignore_index=True)
        return df.drop_duplicates().reset_index(drop=True)
    
    def load_agg_correlation_data(self):
        '''Loads the correlation matrix for each pair (alias, condition)
        as well as the number of cells used to calculate the correlations.'''
        data = []
        for _, comb in self.get_unique_combinations_of_comparisons().iterrows():
            path = Path(self.datasets[comb.alias][comb.phenotype]) / "concordance"
            control, dev = get_managers_from_step_path(path)
            variables = control.get_variables_values_for_aggregation()
            df_agg = shapespace.ShapeSpaceBasic.get_aggregated_df(variables)
            df_agg = df_agg.drop(columns=["structure"]).drop_duplicates().reset_index(drop=True)
            for _, row in df_agg.iterrows():
                df_corr, prefix, df_size = dev.get_mean_correlation_matrix_of_reps(row, return_ncells=True)
                dfs = {"corr": df_corr, "size": df_size}
                data.append((comb.alias, comb.phenotype, prefix, dfs))
        self.data = dict()
        for k1, k2, k3, value in data:
            self.data.setdefault(k1, {}).setdefault(k2, {}).update({k3: value})

    def set_df_of_error_curves(self, df):
        self.df_error = df.copy()        

    def calculate_swaps(self, a1, p1, a2, p2):
        '''By defining the prefix as below we are explicitly using the fact
        that matched-datasets have been processed using a single shape mode
        binned in a single bin.'''
        np.random.seed(42)
        prefix = [k for k in self.data[a1][p1].keys()][0]
        genes = self.data[a1][p1][prefix]["corr"].index
        df_swap = pd.DataFrame(0, index=genes, columns=genes)
        for gid1, gene1 in tqdm(enumerate(genes), total=len(genes), leave=False):
            for gid2, gene2 in enumerate(genes):
                if gid2 >= gid1:
                    n11 = self.data[a1][p1][prefix]["size"].at[gene1, "ncells"]
                    n12 = self.data[a1][p1][prefix]["size"].at[gene2, "ncells"]
                    n21 = self.data[a2][p2][prefix]["size"].at[gene1, "ncells"]
                    n22 = self.data[a2][p2][prefix]["size"].at[gene2, "ncells"]
                    n1 = n11
                    n2 = n22
                    corr1 = self.data[a1][p1][prefix]["corr"].at[gene1, gene2]
                    corr2 = self.data[a2][p2][prefix]["corr"].at[gene1, gene2]
                    std1 = self.get_scale(gene1, gene2, n1)
                    std2 = self.get_scale(gene1, gene2, n2)
                    swap = self.bootstrap(corr1, std1, corr2, std2)
                    df_swap.loc[gene1, gene2] = df_swap.loc[gene2, gene1] = swap
        return df_swap

    def get_scale(self, gene1, gene2, n):
        '''Remove 1 from the ratio to get a proxy for the variation
        around the true value. Limiting scale to 0 in case it goes
        negative due to stochastic fluctuations.'''
        n = 400 if n > 400 else n
        index = (gene1, gene2, n)
        if index in self.df_error.index:
            return self.df_error.at[index, "std"]
        return self.df_error.at[(gene2, gene1, n), "std"]

    @staticmethod
    def bootstrap(x1, scale1, x2, scale2):
        count = 0
        NRUNS = 100
        for r in range(NRUNS):
            x1_new = np.random.normal(loc=x1, scale=abs(x1)*scale1)
            x2_new = np.random.normal(loc=x2, scale=abs(x2)*scale2)
            count += int(~np.logical_xor(x1>=x2, x1_new>=x2_new))
        return 1. - count/NRUNS
    
    def execute(self):
        dfs = {}
        for _, ((a1, p1), (a2, p2)) in self.comparisons.items():
            df_swap = self.calculate_swaps(a1, p1, a2, p2)
            dfs[f"{a1}_{p1}_{a2}_{p2}"] = df_swap
        return dfs
