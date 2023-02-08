from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

'''
from .datasets.coco import COCO
from .datasets.kitti import KITTI
from .datasets.coco_hp import COCOHP
from .datasets.mot import MOT
from .datasets.nuscenes import nuScenes
from .datasets.crowdhuman import CrowdHuman
from .datasets.kitti_tracking import KITTITracking
'''
from src.CNNs_new.lib.dataset.datasets.custom_dataset import CustomDataset

dataset_factory = {
  'custom': CustomDataset,

}


def get_dataset(dataset):
  return dataset_factory[dataset]
