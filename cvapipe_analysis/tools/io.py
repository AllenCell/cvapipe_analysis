import os
import vtk
import errno
import logging
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage import io as skio
from aicsshparam import shtools
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

try:
    from aicsfiles import FileManagementSystem
except: pass

log = logging.getLogger(__name__)

class LocalStagingIO:
    """
    Class that provides functionalities to read and
    write at local_staging.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, control, subfolder=None):
        self.row = None
        self.control = control
        self.subfolder = subfolder

    def get_single_cell_images(self, row, return_stack=False):
        imgs = []
        channel_names = []
        imtypes = ["crop_raw", "crop_seg"]
        for imtype in imtypes:
            if imtype in row:
                path = Path(row[imtype])
                if not path.is_file():
                    path = self.control.get_staging() / f"loaddata/{row[imtype]}"
                reader = AICSImage(path)
                channel_names += reader.channel_names
                img = reader.get_image_data('CZYX', S=0, T=0)
                imgs.append(img)
        try:
            name_dict = eval(row.name_dict)
            channel_names = []
            for imtype in imtypes:
                channel_names += name_dict[imtype]
        except Exception as ex:
            if not channel_names:
                raise ValueError(f"Channel names not found, {ex}")
            else: pass
        imgs = np.vstack(imgs)
        if return_stack:
            return imgs, channel_names
        imgs_dict = {}
        for imtype in imtypes:
            if imtype in row:
                for ch, img in zip(channel_names, imgs):
                    imgs_dict[ch] = img
        return imgs_dict

    def get_abs_path_to_step_manifest(self, step):
        return self.control.get_staging() / f"{step}/manifest.csv"

    def write_compute_features_manifest_from_distributed_results(self):
        df = self.load_step_manifest("loaddata")
        path = self.control.get_staging() / "computefeatures/cell_features"
        df_results = self.load_results_in_single_dataframe(path=path)
        df_results = df_results.set_index('CellId')

        if len(df) != len(df_results):
            df_miss = df.loc[~df.index.isin(df_results.index)]
            log.info("Missing feature for indices:")
            for index in df_miss.index:
                log.info(f"\t{index}")
            log.info(f"Total of {len(df_miss)} indices.")

        manifest = df.merge(df_results, left_index=True, right_index=True, how="outer")
        manifest_path = self.get_abs_path_to_step_manifest("computefeatures")
        manifest.to_csv(manifest_path)
        return manifest

    def load_step_manifest(self, step, clean=False, **kwargs):

        if step == 'computefeatures' and not self.get_abs_path_to_step_manifest(step).is_file():
            df = self.write_compute_features_manifest_from_distributed_results()
        else:
            df = pd.read_csv(
                self.get_abs_path_to_step_manifest(step),
                index_col="CellId", low_memory=False, **kwargs
            )
        
        if clean:
            feats = ['mem_', 'dna_', 'str_'] #TODO: fix the aliases here
            df = df[[c for c in df.columns if not any(s in c for s in feats)]]
        return df

    def read_map_point_mesh(self, alias):
        row = self.row
        path = f"shapemode/avgshape/{alias}_{row.shape_mode}_{row.mpId}.vtk"
        path = self.control.get_staging() / path
        if not path.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return self.read_vtk_polydata(path)

    def read_mean_shape_mesh(self, alias):
        sm = self.control.get_shape_modes()
        mpIdc = self.control.get_center_map_point_index()
        path = f"shapemode/avgshape/{alias}_{sm[0]}_{mpIdc}.vtk"
        path = self.control.get_staging() / path
        if not path.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return self.read_vtk_polydata(path)

    def read_parameterized_intensity(self, index, return_intensity_names=False):
        code, intensity_names = None, []
        path = f"parameterization/representations/{index}.tif"
        path = self.control.get_staging() / path
        if path.is_file():
            code = AICSImage(path)
            intensity_names = code.channel_names
            code = code.data.squeeze()
            if code.ndim == 2:
                code = code.reshape(1, *code.shape)
        if return_intensity_names:
            return code, intensity_names
        return code

    @staticmethod
    def normalize_representations(reps):
        # Expected shape is SCMN
        if reps.ndim != 4:
            raise ValueError(f"Input shape {reps.shape} does not match expected SCMN format.")
        count = np.sum(reps, axis=(-2,-1), keepdims=True)
        reps_norm = np.divide(reps, count, out=np.zeros_like(reps), where=count>0)
        return reps_norm

    def read_normalized_parameterized_intensity_of_alias(self, index, alias):
        reps, names = self.read_parameterized_intensity(index, return_intensity_names=True)
        if reps is None:
            return None
        if reps.ndim == 2:
            reps = reps.reshape(1, *reps.shape)
        rep = reps[names.index(alias)]
        amount = int((rep>0).sum())
        if amount > 0:
            rep[rep>0] = 1.0
            rep /= amount
        return rep

    def read_parameterized_intensity_of_alias(self, index, alias):
        reps, names = self.read_parameterized_intensity(index, return_intensity_names=True)
        if reps is None:
            return None
        if reps.ndim == 2:
            reps = reps.reshape(1, *reps.shape)
        rep = reps[names.index(alias)]
        return rep

    def read_agg_parameterized_intensity(self, row, normalized=False):
        fname = self.get_aggrep_file_name(row)
        if normalized:
            fname = fname.replace(".tif", "_norm.tif")
        path = f"aggregation/repsagg/{fname}"
        path = self.control.get_staging() / path
        if not path.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        code = AICSImage(path)
        code = code.data.squeeze()
        if code.ndim == 2:
            code = code.reshape(1, *code.shape)
        return code

    def load_results_in_single_dataframe(self, path=None):
        ''' Not sure this function is producing a column named index when
        the concordance results are loaded. Further investigation is needed
        here'''
        if path is None:
            path = self.control.get_staging() / self.subfolder
        files = [{"csv": path/f} for f in os.listdir(path)]
        with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
            df = pd.concat(
                tqdm(executor.map(self.load_data_from_csv, files), total=len(files)),
                axis=0, ignore_index=True)
        return df

    def read_corelation_matrix(self, row, return_cellids=False):
        fname = self.get_correlation_matrix_file_prefix(row)

        try:
            corr = skio.imread(f"{self.control.get_staging()}/correlation/values/{fname}.tif")
        except Exception as ex:
            print(f"Correlation matrix {fname} not found.")
            return None
        np.fill_diagonal(corr, np.nan)
        corr_idx = pd.read_csv(f"{self.control.get_staging()}/correlation/values/{fname}.csv", index_col=0)
        df_corr = pd.DataFrame(corr)
        # Include structure name information into the correlation matrix
        df = self.load_step_manifest("loaddata")
        df_corr.columns = pd.MultiIndex.from_tuples([
            (
                df.at[corr_idx.at[c, "CellIds"], "structure_name"],
                corr_idx.at[c, "CellIds"],
                c
            ) for c in df_corr.columns], name=["structure", "CellId", "rank"])
        df_corr.index = pd.MultiIndex.from_tuples([
            (
                df.at[corr_idx.at[c, "CellIds"], "structure_name"],
                corr_idx.at[c,"CellIds"],
                c
            ) for c in df_corr.index], name=["structure", "CellId", "rank"])
        if return_cellids:
            return df_corr, corr_idx
        return df_corr

    #TODO: revist this function (maybe redundant)
    def build_correlation_matrix_of_avg_reps_from_corr_values(self, row, genes=None):
        if genes is None:
            genes = self.control.get_gene_names()
        matrix = np.ones((len(genes), len(genes)))
        for gid1, gene1 in enumerate(genes):
            for gid2, gene2 in enumerate(genes):
                if gid2 > gid1:
                    fname = self.get_correlation_matrix_file_prefix(row, genes=(gene1,gene2))
                    df = pd.read_csv(f"{self.control.get_staging()}/concordance/values/{fname}.csv")
                    matrix[gid1, gid2] = matrix[gid2, gid1] = df.Pearson.values[0]
        return matrix

    def get_correlation_of_mean_reps(self, row, return_ncells=False):
        prefix = self.get_prefix_from_row(row)
        df_corr = pd.read_csv(f"{self.control.get_staging()}/concordance/plots/{prefix}_CORR_OF_AVG_REP.csv", index_col=0)
        if return_ncells:
            CellIds = pd.read_csv(f"{self.control.get_staging()}/correlation/values/{prefix}.csv", index_col=0).CellIds
            df = self.load_step_manifest("preprocessing")
            df = pd.DataFrame(df.loc[CellIds].groupby("structure_name").size(), columns=["ncells"])
            return df_corr, prefix, df
        return df_corr, prefix

    def get_mean_correlation_matrix_of_reps(self, row, return_ncells=False):
        prefix = self.get_prefix_from_row(row)
        df_corr = pd.read_csv(f"{self.control.get_staging()}/concordance/plots/{prefix}_AVG_CORR_OF_REPS.csv", index_col=0)
        if return_ncells:
            CellIds = pd.read_csv(f"{self.control.get_staging()}/correlation/values/{prefix}.csv", index_col=0).CellIds
            df = self.load_step_manifest("preprocessing")
            df = pd.DataFrame(df.loc[CellIds].groupby("structure_name").size(), columns=["ncells"])
            return df_corr, prefix, df
        return df_corr, prefix

    @staticmethod
    def get_correlation_matrix_file_prefix(row, genes=None):#TODO Can be deleted?
        fname = f"{row.aggtype}-{row.alias}-{row.shape_mode}-{row.mpId}"
        if genes is not None:
            fname = f"{row.aggtype}-{row.alias}-{genes[0]}-{genes[1]}-{row.shape_mode}-{row.mpId}"
        return fname

    @staticmethod
    def load_data_from_csv(parameters, use_fms=False):
        if use_fms:
            fms = FileManagementSystem()
            fmsid = parameters['fmsid']
            record = fms.find_one_by_id(fmsid)
            if record is None:
                raise ValueError(f"Record {fmsid} not found on FMS.")
            path = record.path
        else:
            path = Path(parameters['csv'])
            if not path.is_file():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        df = pd.read_csv(path)
        # Backwards compatibility for new DVC data
        df = df.rename(columns={"crop_seg_path": "crop_seg", "crop_raw_path": "crop_raw"})
        return df

    @staticmethod
    def write_ome_tif(path, img, channel_names=None, image_name=None):
        path = Path(path)
        dims = [['X', 'Y', 'Z', 'C', 'T'][d] for d in range(img.ndim)]
        dims = ''.join(dims[::-1])
        name = path.stem if image_name is None else image_name
        OmeTiffWriter.save(img, path, dim_order=dims, image_name=name, channel_names=channel_names)
        return

    @staticmethod
    def read_vtk_polydata(path: Path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput()

    @staticmethod
    def status(idx, output, computed):
        msg = "FAIL"
        if output is not None:
            msg = "COMPLETE" if computed else "SKIP"
        print(f"Index {idx} {msg}. Output: {output}")

    @staticmethod
    def load_csv_file_as_dataframe(fpath):
        df = None
        try:
            df = pd.read_csv(fpath)
        except:
            pass
        return df

    @staticmethod
    def get_prefix_from_row(row):
        fname = []
        for col in ["aggtype",
                    "alias",
                    "structure",
                    "structure1",
                    "structure2",
                    "shape_mode",
                    "mpId"]:
            if col in row:
                val = row[col]
                if not isinstance(val, str):
                    val = str(val)
                fname.append(val)
        return "-".join(fname)

    @staticmethod
    def get_aggrep_file_name(row):
        return f"{row.aggtype}-{row.alias}-{row.structure}-{row.shape_mode}-{row.mpId}.tif"


class DataProducer(LocalStagingIO):
    """
    Functionalities for steps that perform calculations
    in a per row fashion.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, control):
        super().__init__(control)

    def workflow(self):
        pass

    def get_output_file_name(self):
        pass

    def save(self):
        return None

    def set_row(self, row):
        self.row = row
        if "CellIds" in row:
            self.CellIds = self.row.CellIds
            if isinstance(self.CellIds, str):
                self.CellIds = eval(self.CellIds)

    def execute(self, row):
        computed = False
        self.set_row(row)
        path_to_output_file = self.check_output_exist()
        if (path_to_output_file is None) or self.control.overwrite():
            try:
                self.workflow()
                computed = True
                path_to_output_file = self.save()
            except Exception as ex:
                print(f"\n>>>{ex}\n")
                path_to_output_file = None
        self.status(row.name, path_to_output_file, computed)
        return path_to_output_file

    def get_output_file_path(self):
        path = f"{self.subfolder}/{self.get_output_file_name()}"
        return self.control.get_staging() / path

    def check_output_exist(self):
        path_to_output_file = self.get_output_file_path()
        if path_to_output_file.is_file():
            return path_to_output_file
        return None

    def load_single_cell_data(self):
        self.data, self.channels = self.get_single_cell_images(self.row, return_stack="True")
        return

    def align_data(self, force_alignment=False):
        self.angle = np.nan
        self.data_aligned = self.data
        if self.control.should_align() or force_alignment:
            alias_ref = self.control.get_alignment_reference_alias()
            if alias_ref is None:
                raise ValueError("Specify a reference alias for alignment.")
            chn = self.channels.index(self.control.get_channel_from_alias(alias_ref))
            ref = self.data[chn]
            unq = self.control.make_alignment_unique()
            _, self.angle = shtools.align_image_2d(ref, make_unique=unq)
            self.data_aligned = shtools.apply_image_alignment_2d(self.data, self.angle)
        return

    @staticmethod
    def correlate_representations(rep1, rep2):
        pcor = np.corrcoef(rep1.flatten(), rep2.flatten())
        # Returns Nan if rep1 or rep2 is empty.
        return pcor[0, 1]
