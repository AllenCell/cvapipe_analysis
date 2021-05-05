from numpy.core.shape_base import _concatenate_shapes
import vtk
import math
import operator
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import stats as spstats
from scipy import cluster as spcluster
from aicsimageio import AICSImage, writers
from vtk.util.numpy_support import vtk_to_numpy as vtk2np
from cvapipe_analysis.tools import io


class PlotMaker(io.LocalStagingIO):
    """
    Support class. Should not be instantiated directly.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    figs = []
    df = None
    dpi = 72

    def __init__(self, control):
        super().__init__(control)
        self.genes = self.control.get_gene_names()

    def workflow(self):
        pass

    def set_dataframe(self, df, dropna=[]):
        self.df_original = df.copy()
        if dropna:
            self.df_original = self.df_original.dropna(subset=dropna)
        self.df = self.df_original

    def execute(self, display=True, **kwargs):
        if "dpi" in kwargs:
            self.dpi = kwargs["dpi"]
        prefix = None if "prefix" not in kwargs else kwargs["prefix"]
        self.workflow()
        self.save(display, prefix)
        self.figs = []

    def save(self, display=True, prefix=None, full_path_provided=False):
        for (fig, signature) in self.figs:
            if display:
                fig.show()
            else:
                fname = signature
                if hasattr(self, "full_path_provided") and self.full_path_provided:
                    save_dir = self.output_folder
                else:
                    save_dir = self.control.get_staging()/self.subfolder
                save_dir.mkdir(parents=True, exist_ok=True)
                if prefix is not None:
                    fname = f"{prefix}_{signature}"
                fig.savefig(save_dir/f"{fname}.png", facecolor="white")
                plt.close(fig)

    def check_dataframe_exists(self):
        if self.df is None:
            raise ValueError("Please set a dataframe first.")
        return True

    def filter_dataframe(self, filters):
        self.check_dataframe_exists()
        self.df = self.control.get_filtered_dataframe(self.df_original, filters)

    @staticmethod
    def get_correlation_matrix(df, rank):
        n = len(rank)
        matrix = np.empty((n, n))
        matrix[:] = np.nan
        names1 = df.structure1.unique()
        names2 = df.structure2.unique()
        for s1, name1 in enumerate(rank):
            for s2, name2 in enumerate(rank):
                if (name1 in names1) and (name2 in names2):
                    indexes = df.loc[(df.structure1 == name1) &
                                     (df.structure2 == name2)].index
                    matrix[s1, s2] = df.at[indexes[0], "Pearson"]
        return matrix

    @staticmethod
    def get_dataframe_desc(df):
        desc = "-".join([str(sorted(df[f].unique()))
                         for f in ["alias", "shape_mode", "mpId"]])
        desc = desc.replace("[", "").replace("]", "").replace("'", "")
        return desc


class ConcordancePlotMaker(PlotMaker):
    """
    Class for creating concordance heatmaps.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, control, subfolder: Optional[str] = None):
        super().__init__(control)
        self.subfolder = "concordance/plots" if subfolder is None else subfolder

    def workflow(self):
        self.check_dataframe_exists()
        self.check_and_store_parameters()
        self.build_correlation_matrix()
        self.make_heatmap(diagonal=self.multiple_mps)
        if self.multiple_mps:
            self.make_relative_heatmap()
        else:
            self.make_dendrogram()
        return

    def check_and_store_parameters(self):
        mpIds = self.df.mpId.unique()
        if len(mpIds) > 2:
            raise ValueError(f"More than 2 map points found in the dataframe: {mpIds}")
        aliases = self.df.alias.unique()
        if len(aliases) > 1:
            raise ValueError(
                f"Multiples aliases found in the dataframe: {aliases}"
            )
        shape_modes = self.df.shape_mode.unique()
        if len(shape_modes) > 1:
            raise ValueError(
                f"Multiples shape modes found in the dataframe: {shape_modes}"
            )
        self.alias = aliases[0]
        self.shape_mode = shape_modes[0]
        self.mpIds = mpIds
        self.multiple_mps = len(mpIds) > 1
        return

    def build_correlation_matrix(self):
        matrices = []
        for mpId in self.mpIds:
            df_mp = self.df.loc[self.df.mpId == mpId]
            matrices.append(self.get_correlation_matrix(df_mp, self.genes))
        self.matrix = matrices[0]
        if self.multiple_mps:
            matrix_inf = np.tril(matrices[0], -1)
            matrix_sup = np.triu(matrices[1], 1)
            self.matrix = matrix_inf + matrix_sup
        return

    def make_heatmap(self, diagonal=False, cmap="RdBu", **kwargs):
        ns = self.matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=self.dpi)
        ax.imshow(-self.matrix, cmap=cmap, vmin=-1.0, vmax=1.0)
        ax.set_xticks(np.arange(self.matrix.shape[0]))
        ax.set_yticks(np.arange(self.matrix.shape[0]))
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels(self.control.get_structure_names())
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(ns + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(ns + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False)
        if diagonal:
            ax.plot([-0.5, ns - 0.5], [-0.5, ns - 0.5], "k-", linewidth=2)
        ax.set_title(self.get_dataframe_desc(self.df))
        plt.tight_layout()
        prefix = "heatmap" if "prefix" not in kwargs else kwargs["prefix"]
        self.figs.append((fig, f"{prefix}_" + self.get_dataframe_desc(self.df)))
        return

    def get_dataframe_of_center_bin(self, alias, shape_mode):
        cidx = self.control.get_center_map_point_index()
        fltr = {"alias": alias, "shape_mode": shape_mode, "mpId": [cidx]}
        return self.control.get_filtered_dataframe(self.df_original, fltr)

    def make_relative_heatmap(self):
        df = self.get_dataframe_of_center_bin(self.alias, self.shape_mode)
        corr = self.get_correlation_matrix(df, self.genes)
        self.matrix = corr - self.matrix
        np.fill_diagonal(self.matrix, 0)
        self.make_heatmap(diagonal=True, cmap="PRGn", prefix="heatmap_relative")
        return

    def make_dendrogram(self):
        try:
            Z = spcluster.hierarchy.linkage(self.matrix, "average")
        except Exception as ex:
            print(f"Can't generate the dendrogram. Possible NaN in matrix: {ex}")
            return
        Z = spcluster.hierarchy.optimal_leaf_ordering(Z, self.matrix)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        _ = spcluster.hierarchy.dendrogram(
            Z, labels=self.control.get_structure_names(), leaf_rotation=90
        )
        ax.set_title(self.get_dataframe_desc(self.df))
        plt.tight_layout()
        self.figs.append((fig, "dendrogram_" + self.get_dataframe_desc(self.df)))
        return


class StereotypyPlotMaker(PlotMaker):
    """
    Class for ranking and plotting structures according
    to their stereotypy value.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, control, subfolder: Optional[str] = None):
        super().__init__(control)
        self.max_number_of_pairs = 0
        self.subfolder = "stereotypy/plots" if subfolder is None else subfolder

    def set_max_number_of_pairs(self, n):
        self.max_number_of_pairs = n if n > 0 else 0

    def workflow(self):
        self.make_boxplot()

    def make_boxplot(self):
        labels = []
        self.check_dataframe_exists()
        fig, ax = plt.subplots(1, 1, figsize=(7, 8), dpi=self.dpi)
        for sid, gene in enumerate(reversed(self.control.get_gene_names())):
            df_s = self.df.loc[self.df.structure == gene]
            if self.max_number_of_pairs > 0:
                df_s = df_s.sample(n=np.min([len(df_s), self.max_number_of_pairs]))
            y = np.random.normal(size=len(df_s), loc=sid, scale=0.1)
            ax.scatter(df_s.Pearson, y, s=1, c="k", alpha=0.1)
            box = ax.boxplot(
                df_s.Pearson,
                positions=[sid],
                showmeans=True,
                widths=0.75,
                sym="",
                vert=False,
                patch_artist=True,
                meanprops={
                    "marker": "s",
                    "markerfacecolor": "black",
                    "markeredgecolor": "white",
                    "markersize": 5,
                },
            )
            label = f"{self.control.get_structure_name(gene)} (N={len(df_s):04d})"
            labels.append(label)
            box["boxes"][0].set(facecolor=self.control.get_gene_color(gene))
            box["medians"][0].set(color="black")
        ax.set_yticklabels(labels)
        ax.set_xlim(-0.2, 1.0)
        ax.set_xlabel("Pearson correlation coefficient", fontsize=14)
        ax.set_title(self.get_dataframe_desc(self.df))
        ax.grid(True)
        plt.tight_layout()
        self.figs.append((fig, self.get_dataframe_desc(self.df)))


class ShapeSpacePlotMaker(PlotMaker):
    """
    Class for creating plots for shape space.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, control, subfolder: Optional[str] = None):
        super().__init__(control)
        if subfolder is None:
            self.subfolder = "shapemode/pca"
        else:
            self.subfolder = subfolder

    def workflow(self):
        return

    def plot_explained_variance(self, space):
        npcs = self.control.get_number_of_shape_modes()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=self.dpi)
        ax.plot(100 * space.pca.explained_variance_ratio_[:npcs], "-o")
        title = "Cum. variance: (1+2) = {0}%, Total = {1}%".format(
            int(100 * space.pca.explained_variance_ratio_[:2].sum()),
            int(100 * space.pca.explained_variance_ratio_[:].sum()),
        )
        ax.set_xlabel("Component", fontsize=18)
        ax.set_ylabel("Explained variance (%)", fontsize=18)
        ax.set_xticks(np.arange(npcs))
        ax.set_xticklabels(np.arange(1, 1 + npcs))
        ax.set_title(title, fontsize=18)
        plt.tight_layout()
        self.figs.append((fig, "explained_variance"))
        return

    def save_feature_importance(self, space):
        path = f"{self.subfolder}/feature_importance.txt"
        abs_path_txt_file = self.control.get_staging() / path
        print(abs_path_txt_file)
        with open(abs_path_txt_file, "w") as flog:
            for col, sm in enumerate(self.control.iter_shape_modes()):
                exp_var = 100 * space.pca.explained_variance_ratio_[col]
                print(f"\nExplained variance {sm}={exp_var:.1f}%", file=flog)
                '''_PC: raw loading, _aPC: absolute loading and
                _cPC: normalized cummulative loading'''
                pc_name = space.axes.columns[col]
                df_sorted = space.df_feats.sort_values(
                    by=[pc_name.replace("_PC", "_aPC")], ascending=False
                )
                pca_cum_contrib = np.cumsum(
                    df_sorted[pc_name.replace("_PC", "_aPC")].values /
                    df_sorted[pc_name.replace("_PC", "_aPC")].sum()
                )
                pca_cum_thresh = np.abs(pca_cum_contrib - 0.80).argmin()
                df_sorted = df_sorted.head(n=pca_cum_thresh + 1)
                print(df_sorted[[
                    pc_name,
                    pc_name.replace("_PC", "_aPC"),
                    pc_name.replace("_PC", "_cPC"), ]].head(), file=flog
                )
        return

    def plot_pairwise_correlations(self, space, off=0):
        df = space.shape_modes
        nf = len(df.columns)
        if nf < 2:
            return
        npts = df.shape[0]
        cmap = plt.cm.get_cmap("tab10")
        prange = []
        for f in df.columns:
            prange.append(np.percentile(df[f].values, [off, 100 - off]))
        # Create a grid of nfxnf
        fig, axs = plt.subplots(nf, nf, figsize=(2 * nf, 2 * nf), sharex="col",
                                gridspec_kw={"hspace": 0.1, "wspace": 0.1},
                                )
        for f1id, f1 in enumerate(df.columns):
            yrange = []
            for f2id, f2 in enumerate(df.columns):
                ax = axs[f1id, f2id]
                y = df[f1].values
                x = df[f2].values
                valids = np.where((
                    (y > prange[f1id][0]) &
                    (y < prange[f1id][1]) &
                    (x > prange[f2id][0]) &
                    (x < prange[f2id][1])))
                if f2id < f1id:
                    xmin = x[valids].min()
                    xmax = x[valids].max()
                    ymin = y[valids].min()
                    ymax = y[valids].max()
                    yrange.append([ymin, ymax])
                    ax.plot(x[valids], y[valids], ".",
                            markersize=2, color="black", alpha=0.8)
                    ax.plot([xmin, xmax], [xmin, xmax], "--")
                    if f2id:
                        plt.setp(ax.get_yticklabels(), visible=False)
                        ax.tick_params(axis="y", which="both", length=0.0)
                    if f1id < nf - 1:
                        ax.tick_params(axis="x", which="both", length=0.0)
                # Add annotations on upper triangle
                elif f2id > f1id:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis="x", which="both", length=0.0)
                    ax.tick_params(axis="y", which="both", length=0.0)
                    pearson, p_pvalue = spstats.pearsonr(x, y)
                    spearman, s_pvalue = spstats.spearmanr(x, y)
                    ax.text(0.05, 0.8, f"Pearson: {pearson:.2f}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                    ax.text(0.05, 0.6, f"P-value: {p_pvalue:.1E}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                    ax.text(0.05, 0.4, f"Spearman: {spearman:.2f}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                    ax.text(0.05, 0.2, f"P-value: {s_pvalue:.1E}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                # Single variable distribution at diagonal
                else:
                    ax.set_frame_on(False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis="y", which="both", length=0.0)
                    ax.hist(x[valids], bins=16, density=True, histtype="stepfilled",
                            color="white", edgecolor="black", label="Complete",
                            )
                    ax.hist(x[valids], bins=16, density=True, histtype="stepfilled",
                            color=cmap(0), alpha=0.2, label="Incomplete",
                            )
                if f1id == nf - 1:
                    ax.set_xlabel(f2, fontsize=7)
                if not f2id and f1id:
                    ax.set_ylabel(f1, fontsize=7)
            if yrange:
                ymin = np.min([ymin for (ymin, ymax) in yrange])
                ymax = np.max([ymax for (ymin, ymax) in yrange])
                for f2id, f2 in enumerate(df.columns):
                    ax = axs[f1id, f2id]
                    if f2id < f1id:
                        ax.set_ylim(ymin, ymax)

        # Global annotation
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", top=False,
                        bottom=False, left=False, right=False)
        plt.title(f"Total number of points: {npts}", fontsize=24)

        self.figs.append((fig, "pairwise_correlations"))
        return


class ShapeModePlotMaker(PlotMaker):
    """
    Class for creating plots for shape mode step.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, control, subfolder: Optional[str] = None):
        super().__init__(control)
        if subfolder is None:
            self.subfolder = "shapemode/avgshape"
        else:
            self.subfolder = subfolder

    def workflow(self):
        return

    def animate_contours(self, contours, prefix):
        hmin, hmax, vmin, vmax = self.control.get_plot_limits()
        offset = 0.05 * (hmax - hmin)

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.tight_layout()
        plt.close()
        ax.set_xlim(hmin - offset, hmax + offset)
        ax.set_ylim(vmin - offset, vmax + offset)
        ax.set_aspect("equal")
        if not self.control.get_plot_frame:
            ax.axis("off")

        lines = []
        for alias, _ in contours.items():
            color = self.control.get_color_from_alias(alias)
            (line,) = ax.plot([], [], lw=2, color=color)
            lines.append(line)

        def animate(i):
            for alias, line in zip(contours.keys(), lines):
                ct = contours[alias][i]
                mx = ct[:, 0]
                my = ct[:, 1]
                line.set_data(mx, my)
            return lines

        n = self.control.get_number_of_map_points()
        anim = animation.FuncAnimation(
            fig, animate, frames=n, interval=100, blit=True
        )
        fname = self.control.get_staging() / f"{self.subfolder}/{prefix}.gif"
        anim.save(fname, writer="imagemagick", fps=n)
        plt.close("all")
        return

    def load_animated_gif(self, shape_mode, proj):
        fname = self.control.get_staging() / f"{self.subfolder}/{shape_mode}_{proj}.gif"
        image = AICSImage(fname).data.squeeze()
        return image

    def combine_and_save_animated_gifs(self):
        stack = []
        for sm in tqdm(self.control.get_shape_modes()):
            imx = self.load_animated_gif(sm, "x")
            imy = self.load_animated_gif(sm, "y")
            imz = self.load_animated_gif(sm, "z")
            if imx.ndim == 3:
                imx, imy, imz = imx.T, imy.T, imz.T
            img = np.c_[imz, imy, imx]
            img = np.swapaxes(img, 0, 1)
            stack.append(img)
        stack = np.array(stack)
        # Here we deal with RGBA vs grayscale images
        if stack.ndim == 4:
            stack = np.expand_dims(stack, axis=1)
            stack = np.repeat(stack, 4, axis=1)
        stack = np.concatenate(stack[:], axis=-2)[:3]
        stack = np.concatenate([stack[:, :-1], stack[:, ::-1]], axis=1)
        # Reduce the empty space between images
        gaps = stack.min(axis=(0, 1, 3)) < 125
        for _ in range(5):
            gaps[1:-1] = gaps[2:] + gaps[:-2]
        stack = stack[:, :, gaps > 0, :]
        fname = self.control.get_staging() / f"{self.subfolder}/combined.tif"
        with writers.ome_tiff_writer.OmeTiffWriter(fname, overwrite_file=True) as writer:
            writer.save(stack, dimension_order='CZYX')
        return

    @staticmethod
    def find_plane_mesh_intersection(mesh, proj):

        # Find axis orthogonal to the projection of interest
        axis = [a for a in [0, 1, 2] if a not in proj][0]

        # Get all mesh points
        points = vtk2np(mesh.GetPoints().GetData())

        if not np.abs(points[:, axis]).sum():
            raise Exception("Only zeros found in the plane axis.")

        mid = np.mean(points[:, axis])
        '''Set the plane a little off center to avoid undefined intersections.
        Without this the code hangs when the mesh has any edge aligned with the
        projection plane.'''
        mid += 0.75
        offset = 0.1 * np.ptp(points, axis=0).max()

        # Create a vtkPlaneSource
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(4)
        plane.SetYResolution(4)
        if axis == 0:
            plane.SetOrigin(
                mid, points[:, 1].min() - offset, points[:, 2].min() - offset
            )
            plane.SetPoint1(
                mid, points[:, 1].min() - offset, points[:, 2].max() + offset
            )
            plane.SetPoint2(
                mid, points[:, 1].max() + offset, points[:, 2].min() - offset
            )
        if axis == 1:
            plane.SetOrigin(
                points[:, 0].min() - offset, mid, points[:, 2].min() - offset
            )
            plane.SetPoint1(
                points[:, 0].min() - offset, mid, points[:, 2].max() + offset
            )
            plane.SetPoint2(
                points[:, 0].max() + offset, mid, points[:, 2].min() - offset
            )
        if axis == 2:
            plane.SetOrigin(
                points[:, 0].min() - offset, points[:, 1].min() - offset, mid
            )
            plane.SetPoint1(
                points[:, 0].min() - offset, points[:, 1].max() + offset, mid
            )
            plane.SetPoint2(
                points[:, 0].max() + offset, points[:, 1].min() - offset, mid
            )
        plane.Update()
        plane = plane.GetOutput()

        # Trangulate the plane
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(plane)
        triangulate.Update()
        plane = triangulate.GetOutput()

        # Calculate intersection
        intersection = vtk.vtkIntersectionPolyDataFilter()
        intersection.SetInputData(0, mesh)
        intersection.SetInputData(1, plane)
        intersection.Update()
        intersection = intersection.GetOutput()

        # Get coordinates of intersecting points
        points = vtk2np(intersection.GetPoints().GetData())

        # Sorting points clockwise
        # This has been discussed here:
        # https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
        # but seems not to be very efficient. Better version is proposed here:
        # https://stackoverflow.com/questions/57566806/how-to-arrange-the-huge-list-of-2d-coordinates-in-a-clokwise-direction-in-python
        coords = points[:, proj]
        center = tuple(
            map(
                operator.truediv,
                reduce(lambda x, y: map(operator.add, x, y), coords),
                [len(coords)] * 2,
            )
        )
        coords = sorted(
            coords,
            key=lambda coord: (
                -135
                - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center))[::-1])
                )
            )
            % 360,
        )

        # Store sorted coordinates
        # points[:, proj] = coords
        return np.array(coords)

    @staticmethod
    def get_2d_contours(named_meshes, swapxy_on_zproj=False):
        contours = {}
        projs = [[0, 1], [0, 2], [1, 2]]
        if swapxy_on_zproj:
            projs = [[0, 1], [1, 2], [0, 2]]
        for dim, proj in zip(["z", "y", "x"], projs):
            contours[dim] = {}
            for alias, meshes in named_meshes.items():
                contours[dim][alias] = []
                for mesh in meshes:
                    coords = ShapeModePlotMaker.find_plane_mesh_intersection(mesh, proj)
                    if swapxy_on_zproj and dim == 'z':
                        coords = coords[:, ::-1]
                    contours[dim][alias].append(coords)
        return contours

    @staticmethod
    def render_and_save_meshes(meshes, path):
        '''This requires a X server to be available'''
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetOffScreenRendering(1)
        renderWindow.AddRenderer(renderer)
        renderer.SetBackground(1, 1, 1)
        for mesh in meshes:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            renderer.AddActor(actor)
        renderWindow.Render()
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()
        return


class ShapeSpaceMapperPlotMaker(PlotMaker):
    """
    Class for creating plots for shape space mapper.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    full_path_provided = True # See explanation in ShapeSpaceMapper.

    def __init__(self, control, save_dir):
        super().__init__(control)
        self.output_folder = Path(save_dir) / "mapping"

    def workflow(self):
        self.check_dataframe_exists()
        self.plot_mapping_1d()
        return

    def plot_mapping_1d(self):
        df_base = self.df.loc["base"]
        cmap = plt.cm.get_cmap("tab10")
        shape_modes = self.control.get_shape_modes()
        for idx, (ds, df) in enumerate(self.df.groupby(level="dataset", sort=False)):
            if ds != "base":
                fig, axs = plt.subplots(len(shape_modes),1, figsize=(5,1*len(shape_modes)), sharex=True, gridspec_kw={"hspace": 0.5})
                for sm, ax in zip(shape_modes, axs):
                    ax.hist(df_base[sm], bins=32, density=True, alpha=0.5, fc=cmap(0))
                    ax.hist(df[sm],bins=32, density=True, alpha=0.5, fc=cmap(idx))
                    ax.set_xlabel(sm, fontsize=14)
                    ax.set_xlim(-5, 5)
                plt.suptitle(ds, fontsize=18)
                plt.tight_layout()
                self.figs.append((fig, f"mapping_{ds}"))

    def plot_mapping_2d(self):
        cmap = plt.cm.get_cmap("tab10")
        shape_modes = self.control.get_shape_modes()
        argshs = {"bins": 32, "density": True, "histtype": "stepfilled"}
        argsbp = {"vert": False, "showfliers": False, "patch_artist": True}
        grid = {"hspace": 0.0, "wspace": 0.0, 'height_ratios': [
            1, 0.5, 3], 'width_ratios': [3, 1]}
        for sm1, sm2 in tqdm(zip(shape_modes[:-1], shape_modes[1:]), total=len(shape_modes)):
            fig, axs = plt.subplots(3, 2, figsize=(9, 6), gridspec_kw=grid, sharex="col", sharey="row"
                                    )
            for idx in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]:
                axs[idx].axis("off")
            for idx, (ds, df) in enumerate(self.df.groupby(level="dataset", sort=False)):
                axs[0, 0].hist(df[sm1], **argshs, edgecolor="black", fc=[0] * 4, lw=2)
                axs[0, 0].hist(df[sm1], **argshs, color=cmap(idx), alpha=0.8)
                box = axs[1, 0].boxplot(
                    df[sm1], positions=[-idx], widths=[0.6], **argsbp)
                box["boxes"][0].set_facecolor(cmap(idx))
                box["medians"][0].set_color("black")
                axs[2, 1].hist(df[sm2], **argshs, edgecolor="black",
                               fc=[0] * 4, orientation='horizontal', lw=2)
                axs[2, 1].hist(df[sm2], **argshs, color=cmap(idx),
                               alpha=0.8, orientation='horizontal')
                axs[2, 0].scatter(df[sm1], df[sm2], s=5, color=cmap(idx), label=ds)
                axs[2, 0].axhline(y=0.0, color='k', linestyle='--')
                axs[2, 0].axvline(x=0.0, color='k', linestyle='--')
            axs[2, 0].set_xlabel(sm1, fontsize=14)
            axs[2, 0].set_ylabel(sm2, fontsize=14)
            groups, labels = axs[2, 0].get_legend_handles_labels()
            legend = plt.legend(groups, labels, title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
            for lh in legend.legendHandles:
                lh.set_sizes([50])
            plt.tight_layout()
            self.figs.append((fig, f"mapping_{sm1}_{sm2}"))
