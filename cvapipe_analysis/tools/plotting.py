import vtk
import math
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from functools import reduce
from skimage import io as skio
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import lines as pltlines
from scipy import stats as spstats
from scipy import cluster as spcluster
from aicsimageio import AICSImage, writers
from vtk.util import numpy_support as vtknp
from cvapipe_analysis.tools import io, shapespace

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"

class PlotMaker(io.DataProducer):
    """
    Support class. Should not be instantiated directly.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    dpi = 72
    heatmap_vmin = -1.0
    heatmap_vmax = 1.0

    def __init__(self, control):
        super().__init__(control)
        self.df = None
        self.figs = []
        self.dataframes = []
        self.device = io.LocalStagingIO(control)

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
                fig.savefig(save_dir/f"{fname}.pdf", facecolor="white")
                plt.close(fig)
        for (df, signature) in self.dataframes:
            fname = signature
            if hasattr(self, "full_path_provided") and self.full_path_provided:
                save_dir = self.output_folder
            else:
                save_dir = self.control.get_staging()/self.subfolder
            save_dir.mkdir(parents=True, exist_ok=True)
            if prefix is not None:
                fname = f"{prefix}_{signature}"
            df.to_csv(save_dir/f"{fname}.csv")

    def check_dataframe_exists(self):
        if self.df is None:
            raise ValueError("Please set a dataframe first.")
        return True

    def filter_dataframe(self, filters):
        self.check_dataframe_exists()
        self.df = self.control.get_filtered_dataframe(self.df_original, filters)

    def set_heatmap_min_max_values(self, vmin, vmax):
        self.heatmap_vmin = vmin
        self.heatmap_vmax = vmax

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
                    indexes = df.loc[(df.structure1 == name1) & (df.structure2 == name2)].index
                    matrix[s1, s2] = df.at[indexes[0], "Pearson"]
        return matrix

    @staticmethod
    def get_aggregated_matrix_from_df(genes, df_corr):
        matrix = np.zeros((len(genes), len(genes)))
        for gid1, gene1 in enumerate(genes):
            for gid2, gene2 in enumerate(genes):
                if gid2 >= gid1:
                    values = df_corr.loc[(gene1, gene2)].values
                    avg = np.nanmean(values)
                    std = np.nanstd(values)
                    matrix[gid1, gid2] = matrix[gid2, gid1] = avg
        return matrix

    @staticmethod
    def get_dataframe_desc(df):
        desc = "-".join([str(sorted(df[f].unique())) for f in ["alias", "shape_mode", "mpId"]])
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
        self.work_with_avg_reps = False
        self.subfolder = "concordance/plots" if subfolder is None else subfolder
        self.ncells = dict([(gene, None) for gene in control.get_structure_names()])

    def workflow(self):
        suffix = "_AVG_CORR_OF_REPS"
        if self.work_with_avg_reps:
            suffix = "_CORR_OF_AVG_REP"
        self.check_dataframe_exists()
        self.matrix = self.get_agg_correlation_matrix(update_ncells=True)
        # self.matrix = self.normalize_by_diagonal(self.matrix)
        if not self.work_with_avg_reps:
            self.set_heatmap_min_max_values(-0.2, 0.2)
        prefix = self.make_heatmap(self.matrix, suffix=suffix)
        genes = self.control.get_gene_names()
        df_corrs = pd.DataFrame(self.matrix, columns=genes, index=genes)
        self.dataframes.append((df_corrs, prefix))
        self.make_dendrogram(self.matrix, suffix=suffix)
        if self.row.mpId < self.control.get_center_map_point_index():
            self.matrix_mirrored = self.get_agg_correlation_matrix(create_mirrored_matrix=True)
            self.make_heatmap(self.matrix_mirrored, diagonal=True, suffix=f"_mirrored{suffix}")
            self.matrix_relative = self.get_agg_correlation_matrix(create_mirrored_matrix=True, relative_to_center=True)
            if not self.work_with_avg_reps:
                self.set_heatmap_min_max_values(-0.1, 0.1)
            self.make_heatmap(self.matrix_relative, diagonal=True, cmap="PRGn", suffix=f"_relative{suffix}")
        return

    def use_average_representations(self, value):
        self.work_with_avg_reps = value

    @staticmethod
    def normalize_by_diagonal(matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j != i:
                    matrix[i, j] /= matrix[i, i]
        np.fill_diagonal(matrix, 1)
        matrix = 0.5*(matrix+matrix.T)
        return matrix

    def get_agg_correlation_matrix(self, create_mirrored_matrix=False, relative_to_center=False, update_ncells=False):
        row = self.row.copy()
        mpIdc = self.control.get_center_map_point_index()
        if create_mirrored_matrix:
            row.mpId = mpIdc-(row.mpId-mpIdc)
        df_corr = self.device.read_corelation_matrix(row)
        if update_ncells:
            for struct, gene in zip(self.control.get_structure_names(), self.control.get_gene_names()):
                self.ncells[struct] = len(df_corr.loc[gene])
        if df_corr is None:
            return
        genes = self.control.get_gene_names()
        if self.work_with_avg_reps:
            matrix = self.device.build_correlation_matrix_of_avg_reps_from_corr_values(row)
        else:
            matrix = self.get_aggregated_matrix_from_df(genes, df_corr)
        if relative_to_center:
            row_center = row.copy()
            row_center.mpId = mpIdc
            df_corr_center = self.device.read_corelation_matrix(row_center)
            if self.work_with_avg_reps:
                matrix_center = self.device.build_correlation_matrix_of_avg_reps_from_corr_values(row_center)
            else:
                matrix_center = self.get_aggregated_matrix_from_df(genes, df_corr_center)
            matrix =  matrix_center - matrix
            self.matrix = matrix_center - self.matrix
        if create_mirrored_matrix:
            matrix_inf = np.tril(self.matrix, -1)
            matrix_sup = np.triu(matrix, 1)
            matrix = matrix_inf + matrix_sup
        return matrix

    def make_heatmap(self, matrix, diagonal=False, cmap="RdBu", **kwargs):
        ns = matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=self.dpi)
        ax.imshow(-matrix, cmap=cmap, vmin=self.heatmap_vmin, vmax=self.heatmap_vmax)
        ax.set_xticks(np.arange(matrix.shape[0]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.get_xaxis().set_ticklabels([])
        names = self.control.get_structure_names()
        for i, name in enumerate(names):
            n = self.ncells[name]
            names[i] = f"{name} (N={n})" if n is not None else name
        ax.get_yaxis().set_ticklabels(names)
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(ns + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(ns + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False)
        prefix = self.device.get_correlation_matrix_file_prefix(self.row)
        if diagonal:
            ax.plot([-0.5, ns - 0.5], [-0.5, ns - 0.5], "k-", linewidth=2)
        if "suffix" in kwargs:
            prefix += kwargs["suffix"]
        ax.set_title(prefix)
        plt.tight_layout()
        self.figs.append((fig, prefix))
        return prefix

    def make_confidence_heatmap(self, matrix, conf_matrix, colors, size_scale=49, dotted=[], hide=[], background_dot=True, markers=None, highlight=[], ec_on=True, **kwargs):

        yxfac = 1 if matrix.shape[0]==matrix.shape[1] else 0.36#3*matrix.shape[1]/matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(8*yxfac, 8), dpi=self.dpi)

        y_labels = matrix.index
        x_labels = matrix.columns
        x_to_num = {xlab:i for i, xlab in enumerate(x_labels)} 
        y_to_num = {ylab:i for i, ylab in enumerate(y_labels)} 

        for ylab, row in matrix.iterrows():
            for xlab, value in row.items():
                if value not in hide:
                    if value not in dotted:
                        if background_dot:
                            ax.scatter(
                                x=x_to_num[xlab],
                                y=y_to_num[ylab],
                                facecolor=(0.8, 0.8, 0.8),
                                s=conf_matrix.at[ylab,xlab] * size_scale,
                                marker='o',
                                ec=(0.8, 0.8, 0.8) if value not in highlight else "red",
                                lw=2
                            )                        
                        ax.scatter(
                            x=x_to_num[xlab],
                            y=y_to_num[ylab],
                            color=colors[value],
                            s=conf_matrix.at[ylab,xlab] * size_scale,
                            marker='s' if markers is None else markers[value],
                            ec="black" if ec_on else None
                        )
                    else:
                        ax.scatter(
                            x=x_to_num[xlab],
                            y=y_to_num[ylab],
                            color="k",
                            s=size_scale,
                            marker='.',
                            ec="black"
                        )
        ns = len(conf_matrix)
        ax.set_ylim(-1, ns+1)
        ax.invert_yaxis()
        ax.set_yticks([y_to_num[v] for v in y_labels])
        names = self.control.get_structure_names()
        for i, name in enumerate(names):
            n = self.ncells[name]
            names[i] = f"{name} (N={n})" if n is not None else name
        ax.set_yticklabels(names)
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.get_xaxis().set_ticklabels([])
        ax.tick_params(which="both", bottom=False, left=False)
        prefix = self.device.get_correlation_matrix_file_prefix(self.row)
        if "suffix" in kwargs:
            prefix += kwargs["suffix"]
        if "xlim" in kwargs:
            ax.set_xlim(*kwargs["xlim"])
        ax.set_title(prefix)
        plt.tight_layout()
        self.figs.append((fig, prefix))
        return prefix

    def make_dendrogram(self, matrix, **kwargs):
        method = "average"
        try:
            Z = spcluster.hierarchy.linkage(matrix, method)
        except Exception as ex:
            print(f"Can't generate the dendrogram. Possible NaN in matrix: {ex}")
            return
        Z = spcluster.hierarchy.optimal_leaf_ordering(Z, matrix)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        _ = spcluster.hierarchy.dendrogram(Z, labels=self.control.get_structure_names(), leaf_rotation=90)
        desc = self.device.get_correlation_matrix_file_prefix(self.row)
        if "suffix" in kwargs:
            desc += kwargs["suffix"]
        ax.set_title(desc)
        plt.tight_layout()
        self.figs.append((fig, "dendrogram_"+desc))
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
        self.matrix = None
        self.extra_values = None
        self.subfolder = "stereotypy/plots" if subfolder is None else subfolder

    def workflow(self):
        self.make_boxplot()
        self.load_data_for_extra_values()
        for matrix, relative in zip([self.matrix, self.matrix_relative], [False, True]):
            vmin, vmax = -0.2, 0.2
            if relative:
                vmin, vmax = -0.1, 0.1
            self.set_heatmap_min_max_values(vmin, vmax)
            self.make_heatmap(matrix=matrix, relative=relative)

    def set_extra_values(self, values):
        self.extra_values = values

    def load_data_for_extra_values(self):
        self.matrix, self.matrix_relative = None, None
        if self.extra_values is None:
            return
        cols = []
        heatmap = []
        row = self.row.copy()
        genes = self.control.get_gene_names()
        for k, vs in self.extra_values.items():
            for v in vs:
                row[k] = v
                df_corr = self.device.read_corelation_matrix(row)
                df_corr_agg = self.get_aggregated_matrix_from_df(genes, df_corr)
                heatmap.append(np.diag(df_corr_agg))
                cols.append(f"{k}={v}")
        self.matrix = np.array(heatmap).T

        df_stereo = pd.DataFrame(self.matrix, columns=cols, index=self.control.get_gene_names())
        prefix = self.device.get_correlation_matrix_file_prefix(self.row)
        prefix += "".join([f"{k}_"+"_".join([str(v) for v in self.extra_values[k]]) for k in self.extra_values.keys()])
        self.dataframes.append((df_stereo, prefix))
        # Relative heatmap
        df_stereo_relative = df_stereo.copy()
        center_col = [f"{k}={self.control.get_center_map_point_index()}" for k in self.extra_values.keys()][0]
        for col in cols:
            df_stereo_relative[col] = df_stereo[center_col]-df_stereo_relative[col]
        self.matrix_relative = df_stereo_relative.values
        self.dataframes.append((df_stereo_relative, prefix+"_relative"))
        return

    def make_boxplot(self):

        df_corr = self.device.read_corelation_matrix(self.row)
        if df_corr is None:
            return

        labels = []
        self.check_dataframe_exists()
        fig, ax = plt.subplots(1, 1, figsize=(7, 8), dpi=self.dpi)
        for sid, gene in enumerate(reversed(self.control.get_gene_names())):

            values = df_corr.loc[gene, gene].values
            ncells = values.shape[0]
            values = values[np.triu_indices(ncells, k=1)]

            np.random.seed(42)
            x = np.random.choice(values, np.min([ncells, 1024]), replace=False)
            y = np.random.normal(size=len(x), loc=sid, scale=0.1)
            ax.scatter(x, y, s=1, c="k", alpha=0.1)
            box = ax.boxplot(
                values,
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
            label = f"{self.control.get_structure_name(gene)} (N={ncells:04d})"
            labels.append(label)
            box["boxes"][0].set(facecolor=self.control.get_gene_color(gene))
            box["medians"][0].set(color="black")
        ax.set_yticklabels(labels)
        ax.set_xlim(-0.2, 1.0)
        ax.set_xlabel("Pearson correlation coefficient", fontsize=14)
        ax.set_title(self.device.get_correlation_matrix_file_prefix(self.row))
        ax.grid(True)
        plt.tight_layout()
        self.figs.append((fig, self.device.get_correlation_matrix_file_prefix(self.row)))

    def make_heatmap(self, matrix, relative=False):
        if matrix is None:
            return
        nsy, nsx = matrix.shape
        cmap = "PRGn" if relative else "RdBu"
        fig, ax = plt.subplots(1, 1, figsize=(8, 16), dpi=self.dpi)
        ax.imshow(-matrix, cmap=cmap, vmin=self.heatmap_vmin, vmax=self.heatmap_vmax)
        ax.set_xticks(np.arange(matrix.shape[0]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels(self.control.get_structure_names())
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(nsx + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(nsy + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False)
        prefix = self.device.get_correlation_matrix_file_prefix(self.row)
        prefix += "".join([f"{k}_"+"_".join([str(v) for v in self.extra_values[k]]) for k in self.extra_values.keys()])
        if relative:
            prefix += "_relative"
        ax.set_title(prefix)
        plt.tight_layout()
        self.figs.append((fig, prefix))
        return


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
        if not self.control.get_plot_frame():
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
        anim.save(fname, fps=n)
        plt.close("all")
        return

    def load_animated_gif(self, shape_mode, proj):
        fname = self.control.get_staging() / f"{self.subfolder}/{shape_mode}_{proj}.gif"
        image = AICSImage(fname).data.squeeze()
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def combine_and_save_animated_gifs(self):
        stack = []
        for sm in tqdm(self.control.get_shape_modes()):
            imx = self.load_animated_gif(sm, "x")
            imy = self.load_animated_gif(sm, "y")
            imz = self.load_animated_gif(sm, "z")
            img = np.concatenate([imz, imy, imx], axis=-2)
            stack.append(img)
        stack = np.array(stack)
        stack = np.concatenate(stack[:], axis=-3)
        stack = np.rollaxis(stack, -1, 1)
        fname = self.control.get_staging() / f"{self.subfolder}/combined.tif"
        writers.ome_tiff_writer.OmeTiffWriter.save(stack, fname, overwrite_file=True, dim_order="CZYX")
        return

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

    grouping = None
    full_path_provided = True # See explanation in ShapeSpaceMapper.

    def __init__(self, control, save_dir):
        super().__init__(control)
        self.output_folder = Path(save_dir) / "mapping"

    def workflow(self):
        self.check_dataframe_exists()
        self.plot_distance_vs_ncells()
        self.plot_mapping_1d()
        self.plot_nn_distance_distributions()
        self.plot_self_distance_distributions()
        return

    def set_grouping(self, grouping):
        self.grouping = grouping

    @staticmethod
    def comparative_hists(df1, df2, title, bin_edges, display_both=True, ymax=1):
        nc = len(df1.columns)
        args = {"bins": bin_edges, "density": True}
        fig, axs = plt.subplots(1, nc, figsize=(1.5*nc, 1.5), sharex=False, gridspec_kw={"wspace": 0.2})
        axs = [axs] if len(axs)==1 else axs
        for sm, ax in zip(df1.columns, axs):
            ax.set_frame_on(False)
            if display_both:
                ax.hist(df1[sm], **args, alpha=0.4, fc="gray")
                ax.hist(df2[sm], **args, histtype="step", linewidth=1, edgecolor="#FF3264")
            else:
                ax.hist(df1[sm], **args, histtype="step", linewidth=1, edgecolor="black")
            # ax.set_xlabel(sm, fontsize=12)
            ax.set_ylim(0, ymax)
            ax.set_xlim(int(bin_edges[0]-1), int(bin_edges[-1]+1))
            ax.get_xaxis().tick_bottom()
            ax.set_xticks([-2, 0, 2])
            ax.axes.get_yaxis().set_visible(False)
            xmin, xmax = ax.get_xaxis().get_view_interval()
            ymin, ymax = ax.get_yaxis().get_view_interval()
            ax.add_artist(pltlines.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
        # plt.tight_layout()
        return fig

    def plot_mapping_1d(self, display_both=True, ymax=1):
        df_base = self.df.loc["base"]
        sms = self.control.get_shape_modes()
        bin_centers = self.control.get_map_points()
        binw = 0.5*np.diff(bin_centers).mean() if len(bin_centers) > 1 else 1
        bin_edges = np.unique([(b-binw, b+binw) for b in bin_centers])
        for dsname, df in self.df.groupby(level="dataset", sort=False):
            if dsname != "base":
                fig = self.comparative_hists(df_base[sms], df[sms], dsname, bin_edges, ymax=ymax)
                self.figs.append((fig, f"mapping_{dsname}"))
                # Same plots for matching pairs
                ds_ids = df.loc[df.Match==True].index
                NNCellIds = df.loc[ds_ids].NNCellId.values
                bs_ids = [(sname, nnid) for ((_, sname, _), nnid) in zip(ds_ids, NNCellIds)]
                bs_ids = pd.MultiIndex.from_tuples(bs_ids).drop_duplicates()
                fig = self.comparative_hists(df_base.loc[bs_ids, sms], df.loc[ds_ids, sms], dsname, bin_edges, display_both, ymax=ymax)
                self.figs.append((fig, f"mapping_{dsname}_match"))

    def plot_mapping_2d(self):
        cmap = plt.cm.get_cmap("tab10")
        shape_modes = self.control.get_shape_modes()
        argshs = {"bins": 32, "density": True, "histtype": "stepfilled"}
        argsbp = {"vert": False, "showfliers": False, "patch_artist": True}
        grid = {"hspace": 0.0, "wspace": 0.0, 'height_ratios': [
            1, 0.5, 3], 'width_ratios': [3, 1]}
        for sm1, sm2 in tqdm(zip(shape_modes[:-1], shape_modes[1:]), total=len(shape_modes)):
            fig, axs = plt.subplots(3, 2, figsize=(9, 6), gridspec_kw=grid, sharex="col", sharey="row")
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

    def plot_nn_distance_distributions(self):
        if self.grouping is None:
            return
        cmap = plt.cm.get_cmap("tab10")
        hargs = {"bins": 32, "linewidth": 3, "density": True, "histtype": "step"}
        for idx, (ds, df) in enumerate(self.df.groupby(level="dataset", sort=False)):
            if ds != "base":
                fig, ax = plt.subplots(1,1, figsize=(5,4))
                ax.hist(df.Dist, bins=32, density=True, alpha=0.5, fc="black", label="All")
                # ax.axvline(df.DistThresh.values[0], color="black")
                for gid, (group, snames) in enumerate(self.grouping.items()):
                    str_available = df.index.get_level_values(level="structure_name")
                    df_group = df.loc[(ds, [s for s in snames if s in str_available]), ]
                    ax.hist(df_group.Dist, **hargs, edgecolor=cmap(gid), label=group)
                ax.set_xlabel("NN Distance", fontsize=14)
                ax.set_xlim(0, 8)
                plt.suptitle(ds, fontsize=14)
                plt.legend()
                plt.tight_layout()
                self.figs.append((fig, f"nndist_{ds}"))

    def plot_self_distance_distributions(self):
        cmap = plt.cm.get_cmap("tab10")
        hargs = {"bins": 32, "linewidth": 3, "density": True, "histtype": "step"}
        for idx, (ds, df) in enumerate(self.df.groupby(level="dataset", sort=False)):
            if ds != "base":
                fig, ax = plt.subplots(1,1, figsize=(5,4))
                ax.hist(df.Dist, bins=32, density=True, alpha=0.5, fc="black", label="nn")
                df = df.dropna(subset=["SelfDist"])
                if len(df):
                    ax.hist(df.SelfDist, **hargs, edgecolor="k", label="self")
                ax.set_xlabel("Distance", fontsize=14)
                ax.set_xlim(0, 8)
                plt.suptitle(ds, fontsize=14)
                plt.legend()
                plt.tight_layout()
                self.figs.append((fig, f"nndist_self_{ds}"))

    def plot_distance_vs_ncells(self):
        for idx, (ds, df) in enumerate(self.df.groupby(level="dataset", sort=False)):
            if ds != "base":
                grid = {'height_ratios': [6, 1]}
                fig, (ax, lg) = plt.subplots(2,1, figsize=(8,4), gridspec_kw=grid)
                for sid, sname in enumerate(self.control.get_gene_names()):
                    y = df.loc[(ds, sname), "Dist"].values
                    ax.axhline(y=1.0, color="black", alpha=0.2, linestyle="--")
                    ax.plot([y.size, y.size], [np.median(y)-y.std(), np.median(y)+y.std()], color="black", alpha=0.5)
                    ax.scatter(y.size, np.median(y), s=50, color=self.control.get_gene_color(sname))
                    lg.scatter(sid, 1, color=self.control.get_gene_color(sname), s=50)
                lg.set_yticks([])
                lg.set_xticks(np.arange(len(self.control.get_gene_names())))
                lg.set_xticklabels(self.control.get_gene_names(), rotation=90)
                for k in lg.spines:
                    lg.spines[k].set_visible(False)
                ax.set_ylim(0, 2)
                ax.set_ylabel("NN Distance (units of std)", fontsize=12)
                ax.set_xlabel("Number of cells", fontsize=14)
                plt.suptitle(ds, fontsize=14)
                plt.tight_layout()
                self.figs.append((fig, f"nndist_ncells_{ds}"))
