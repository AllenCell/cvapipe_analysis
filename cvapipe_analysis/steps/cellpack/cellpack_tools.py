import numpy as np
from skimage import measure as skmeasure
from aicsimageio import writers

from cvapipe_analysis.tools import io

class ObjectCollector(io.DataProducer):
    """
    Desc
    
    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    structures = None
    subfolder = 'cellpack/data'
    
    def __init__(self, config):
        super().__init__(config)
        self.channel = config['cellpack']['channel']
    
    def load_segmentation(self):
        reader = general.LocalStagingReader(self.config, self.row)
        segs = reader.get_single_cell_images('crop_seg')
        self.seg = segs[self.channel]
        return

    def collect_segmented_objects(self, row):
        objs = []
        self.row = row
        self.load_segmentation()
        seg_labeled = skmeasure.label(self.seg)
        max_label = seg_labeled.max()
        for label in range(1, max_label+1):
            coords = np.where(seg_labeled==label)
            coords = np.array(coords).T
            center = coords.min(axis=0) + 0.5*np.ptp(coords, axis=0)
            coords -= center.astype(coords.dtype)
            objs.append(coords)
        #bbox = [np.ptp(obj, axis=0) for obj in objs]
        #bbox = np.array(bbox).max(axis=0)
        return objs

    def save_img(self, img, prefix):
        save_as = self.abs_path_local_staging/f"{self.subfolder}/{prefix}.tif"
        save_as.parent.mkdir(parents=True, exist_ok=True)
        with writers.ome_tiff_writer.OmeTiffWriter(save_as, overwrite_file=True) as writer:
            writer.save(img, dimension_order = 'CZYX', image_name = save_as.stem)
    
    @staticmethod
    def pack_objs(objs):
        bbox = [np.ptp(obj, axis=0) for obj in objs]
        bbox = 5 + np.array(bbox).max(axis=0)
        img = np.zeros((len(objs),*bbox), dtype=np.uint8)
        centroid = (0.5*bbox).reshape(1, *bbox.shape).astype(np.int)
        for o, obj in enumerate(objs):
            coords = obj + centroid
            img[o, coords[:,0], coords[:,1], coords[:,2]] = 255
        return img
