import os
import sys
import json
import datetime
import numpy as np
import pathlib
import skimage.draw



'''전역변수 : ROOT_DIR'''

# Root directory of the project
'''ROOT_DIR=tensor_code/FASHION'''
ROOT_DIR=os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library  
from mrcnn.config import Config
from mrcnn import model as modellib, utils


'''일단 load할 weight가 없음'''
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")




class FashionConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Fashion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + Category of Clothes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9







############################################################
#  Dataset
############################################################

class FashionDataset(utils.Dataset):


    def __init__(self):
        super().__init__()

        self.class_dict={1:'short sleeve top',2:'long sleeve top',3:'short sleeve outwear',4:'long sleeve outwear',5:'vest',6:'sling',\
                        7:'shorts',8:'trousers',9:'skirt',10:'short sleeve dress',11:'long sleeve dress',12:'vest dress',13:'sling dress'}


    def load_fashion(self, parent_dir:str, subset_idx:str):
        """Load a subset of the Balloon dataset.
        parent_dir:  C:\tensor_code\fashion\dataset\deepfashion2\train
        subset_idx : '0'
        subset: train_anno_0
        """

        '''
        아래 부분 상속
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        add_classes. 13개의 fashion dataset+ BG(BG는 이미 되어있음 : 0번)
        '''
        for key,value in self.class_dict.items():
            self.add_class('deepfashion',key,value)
        

        # Train or validation dataset?
        #assert subset in ["train", "val"]

        parent_dir=pathlib.Path(parent_dir)

        json_child_name='train_anno_'+subset_idx
        json_dir=parent_dir/json_susbet_iter
        
        json_files_iter=pathlib.Path(json_subset_dir).glob('*')
        json_files=[i for i in json_files_iter]
    
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
       
        image_info = self.image_info[image_id]

        assert image_info["source"] == "deepfashion"

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
