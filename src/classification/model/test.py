# DeepLerning libs
import tensorflow as tf
from tensorflow.keras import models
from . import const
from . import Network
from .Network import BaseNetwork 
from . import util
from . import DA
from . import DA_setting

# enum
from main.project_type import ProjectType

# other libs
import csv
import glob
import json
import numpy as np
import os
import random
import shutil
import sys

from main import file_action

from main.log import get_logger

logger = get_logger(__name__)

def main(self,
         model_root:str,
         project,
         model,
         num_classes:int,
         image_type:str,
         architecture:str,
         predict_data:list,
         class_list:list,
         weights_file_path:str):

    logger.debug(f"model_root        :{model_root}")
    logger.debug(f"project           :{project}")
    logger.debug(f"model             :{model}")
    logger.debug(f"num_classes       :{num_classes}")
    logger.debug(f"image_type        :{image_type}")
    logger.debug(f"architecture      :{architecture}")
    logger.debug(f"predict_data count:{len(predict_data)}")
    logger.info(f"Inside Test Module")

    logger.debug(f"predict_data: {predict_data}")

    project_type = ProjectType(project.project_type)

    # test tree
    test_path = os.path.join(model_root, "test")
    test_image_path = os.path.join(test_path, "images")
    test_result_path = os.path.join(test_path, "result")

    os.makedirs(test_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(test_result_path, exist_ok=True)

    
    # GPU setting
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
        logger.info(f"GPU Selected: {physical_devices[0]}")

    # network setting
    size = [224,224] #TODO
    net = Network.Network(channel=3, classes=num_classes, network=architecture, input_size=size,image_type=image_type)
    logger.info(f"weights loaded: {weights_file_path}")
    net.model.load_weights(weights_file_path)

    # run predict
    pred_probs = net.predict(predict_data)

    # result modeling
    if architecture == const.GOOGLE_NET:
        pred_probs = pred_probs[2] # loss2

    #pred_prob = np.asarray(pred_prob)
    preds = pred_probs.argmax(axis=1)
    
    files = [] 
    labels = np.empty([len(predict_data), num_classes])
    for idx ,data in enumerate (predict_data):
        files.append(data[0])
        labels[idx] = data[1]

    labels = np.argmax(labels, axis=1)

    logger.debug("preds: {}".format(preds))
    logger.debug("pred_probs: {}".format(pred_probs))
    logger.debug("label: {}".format(labels))

    return preds, pred_probs, labels
