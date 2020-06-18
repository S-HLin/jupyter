import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io

# import sys
# sys.path
# Import Mask RCNN
ROOT_DIR = os.path.abspath("./maskrcnn/")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import base64
import cv2

# import Ipynb_importer
# from samples.food import food
DEVICE = "/cpu:0"
TEST_MODE = "inference"
IMAGE_DIR = '/dataDisk/myfloder/jupyter/Mask_RCNN-master/datasets/food/val2/'
MODEL_DIR = '/dataDisk/myfloder/h5/maskrcnn_snapshots/'
weights_path = '/dataDisk/myfloder/h5/maskrcnn_snapshots/food20191127T1334/mask_rcnn_food_0500.h5'
class_names = ['BG', '蛋', '豬肉', '雞肉', '牛肉', '魚']

class objectConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "food"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

#object_DIR = os.path.join(ROOT_DIR, "datasets/food")
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(objectConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#config = InferenceConfig()
#config.display()

class demo_maskrcnn():

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", 
                                  model_dir=MODEL_DIR,
                                  config=InferenceConfig())
    # Or, load the last model you trained
    #weights_path = model.find_last()

    # Load weights
    #print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    #model.keras_model.save('/dataDisk/myfloder/h5/mymaskmodel.h5')
    #model.keras_model.summary()

    def base64_to_image(self,base64_code): 
        # base64解码
        img_data = base64.b64decode(base64_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) 

        imageBGR = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
        return imageBGR

    def ndarray_to_base64(self,img_ndarray):
        retval, buffer = cv2.imencode('.png', img_ndarray)
        img_str = base64.b64encode(buffer)
        img_str = img_str.decode()
        return img_str

    def predictimg(self,image):
        if image.shape[-1] == 4:    
            image = image[...,:3]

        # Run detection
        results = self.model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        ax , outimg = visualize.display_instances(image, 
                                    r['rois'], 
                                    r['masks'], 
                                    r['class_ids'], 
                                    class_names, 
                                    r['scores'])    
        return ax , outimg

