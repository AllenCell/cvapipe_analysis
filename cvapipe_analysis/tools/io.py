import os
import vtk
import errno
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from aicsshparam import shtools
from aicsimageio import AICSImage, writers


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
        imtypes = [k for k in eval(row.name_dict).keys()]
        channel_names = [v for k, v in eval(row.name_dict).items()]
        channel_names = [ch for names in channel_names for ch in names]
        for imtype in imtypes:
            if imtype in row:
                path = row[imtype]
                if str(self.control.get_staging()) not in path:
                    path = self.control.get_staging() / f"loaddata/{row[imtype]}"
                img = AICSImage(path).data.squeeze()
                imgs.append(img)
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

    def load_step_manifest(self, step, clean=False):
        df = pd.read_csv(
            self.get_abs_path_to_step_manifest(step),
            index_col="CellId", low_memory=False
        )
        if clean:
            feats = ['mem_', 'dna_', 'str_']
            df = df[[c for c in df.columns if not any(s in c for s in feats)]]
        return df

    def read_map_point_mesh(self, alias):
        row = self.row
        path = f"shapemode/avgshape/{alias}_{row.shape_mode}_{row.mpId}.vtk"
        path = self.control.get_staging() / path
        if not path.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return self.read_vtk_polydata(path)

    def read_parameterized_intensity(self, index, return_intensity_names=False):
        path = f"parameterization/representations/{index}.tif"
        path = self.control.get_staging() / path
        if not path.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        code = AICSImage(path)
        intensity_names = code.get_channel_names()
        code = code.data.squeeze()
        if return_intensity_names:
            return code, intensity_names
        return code

    def read_agg_parameterized_intensity(self, row):
        path = f"aggregation/repsagg/{self.get_aggrep_file_name(row)}"
        path = self.control.get_staging() / path
        if not path.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        code = AICSImage(path)
        code = code.data.squeeze()
        return code

    def load_results_in_single_dataframe(self):
        ''' Not sure this function is producing a column named index when
        the concordance results are loaded. Further investigation is needed
        here'''
        path_to_output_folder = self.control.get_staging() / self.subfolder
        files = [path_to_output_folder / f for f in os.listdir(path_to_output_folder)]
        with concurrent.futures.ProcessPoolExecutor(self.control.get_ncores()) as executor:
            df = pd.concat(
                tqdm(executor.map(self.load_csv_file_as_dataframe, files), total=len(files)),
                axis=0, ignore_index=True)
        return df

    @staticmethod
    def write_ome_tif(path, img, channel_names=None, image_name=None):
        path = Path(path)
        dims = [['X', 'Y', 'Z', 'C', 'T'][d] for d in range(img.ndim)]
        dims = ''.join(dims[::-1])
        name = path.stem if image_name is None else image_name
        with writers.ome_tiff_writer.OmeTiffWriter(path, overwrite_file=True) as writer:
            writer.save(
                img, dimension_order=dims, image_name=name, channel_names=channel_names)
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
        self.data, self.channels = self.get_single_cell_images(
            self.row, return_stack="True")
        return

    def align_data(self):
        self.angle = np.nan
        self.data_aligned = self.data
        if self.control.should_align():
            alias_ref = self.control.get_alignment_reference_alias()
            if alias_ref is None:
                raise ValueError("Specify a reference alias for alignment.")
            chn = self.channels.index(self.control.get_channel_from_alias(alias_ref))
            ref = self.data[chn]
            unq = self.control.make_alignment_unique()
            _, self.angle = shtools.align_image_2d(ref, unq)
            self.data_aligned = shtools.apply_image_alignment_2d(self.data, self.angle)
        return

    @staticmethod
    def correlate_representations(rep1, rep2):
        pcor = np.corrcoef(rep1.flatten(), rep2.flatten())
        # Returns Nan if rep1 or rep2 is empty.
        return pcor[0, 1]
