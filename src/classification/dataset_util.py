from deepeye.settings import MINIMUM_IMAGES
from .models import ClassData, DatasetData, ImageData
from .models import TestResult
from main.dataset_data_type import DatasetDataType

import xml.etree.ElementTree as ET
import os
import json
import glob
import csv
import shutil
import random
import numpy as np
import random
import time

from main import file_action

from main.log import get_logger

logger = get_logger(__name__)

def create_image_list(dataset, val_ratio, test_ratio, model=None):
    unique_id = get_unique_id()

    val_par = float(val_ratio) / 100
    test_par = float(test_ratio) / 100

    all_image_data_list = ImageData.objects.filter(dataset=dataset)
    n_image_data = len(all_image_data_list)

    class_data_list = ClassData.objects.filter(dataset=dataset).order_by('class_id')

    train_id_list = []
    val_id_list = []
    test_id_list = []

    for class_data in class_data_list:
        image_data_list_by_class = ImageData.objects.filter(dataset=dataset, class_data=class_data)

        n = len(image_data_list_by_class)

        n_val = int(n * val_par)
        if n_val < 1:
            n_val = 1
        
        n_test = int(n * test_par)
        if n_test < 1:
            n_test = 1

        n_train = n - (n_val + n_test)
        if n_train < 1:
            raise ValueError('Not enough data. (class {})'.format(class_data.name))

        logger.debug('{} n: {} train: {}, val: {}, test:{}'.format(class_data.name, n, n_train, n_val, n_test))

        train_id_list.extend([image_data.id for image_data in image_data_list_by_class[:n_train]])
        val_id_list.extend([image_data.id for image_data in image_data_list_by_class[n_train:(n_train + n_val)]])
        test_id_list.extend([image_data.id for image_data in  image_data_list_by_class[(n_train + n_val):]])

    random.seed(0)
    random.shuffle(train_id_list)
    random.shuffle(val_id_list)
    random.shuffle(test_id_list)

    logger.debug('train_id_list: {}'.format(train_id_list))
    logger.debug('val_id_list: {}'.format(val_id_list))
    logger.debug('test_id_list: {}'.format(test_id_list))

    for i, id in enumerate(train_id_list):
        image_data = ImageData.objects.get(pk=id)
        data = DatasetData(data_id=i, dataset_data_type=DatasetDataType.Train, image_data=image_data, model=model, dataset=dataset, unique_id=unique_id)
        data.save()

    for i, id in enumerate(val_id_list):
        image_data = ImageData.objects.get(pk=id)
        data = DatasetData(data_id=i, dataset_data_type=DatasetDataType.Validation, image_data=image_data, model=model, dataset=dataset, unique_id=unique_id)
        data.save()

    for i, id in enumerate(test_id_list):
        image_data = ImageData.objects.get(pk=id)
        data = DatasetData(data_id=i, dataset_data_type=DatasetDataType.Test, image_data=image_data, model=model, dataset=dataset, unique_id=unique_id)
        data.save()

    return unique_id

def get_dataset_list(project, dataset, dataset_data_type, unique_id=None):
    dataset_data_list = []

    if type(dataset_data_type) is list:
        if unique_id is None:
            dataset_data = DatasetData.objects.filter(dataset=dataset, dataset_data_type__in=dataset_data_type).order_by("data_id")
        else:
            dataset_data = DatasetData.objects.filter(dataset=dataset, dataset_data_type__in=dataset_data_type, unique_id=unique_id).order_by("data_id")
    else:
        if unique_id is None:
            dataset_data = DatasetData.objects.filter(dataset=dataset, dataset_data_type=dataset_data_type).order_by("data_id")
        else:
            dataset_data = DatasetData.objects.filter(dataset=dataset, dataset_data_type=dataset_data_type, unique_id=unique_id).order_by("data_id")

    for data in dataset_data:
        on_hot_class = np.zeros((1, dataset.classes), dtype=np.int64)
        on_hot_class[0, data.image_data.class_data.class_id] = 1

        dataset_data_list.append([file_action.get_image_local_full_path(dataset, data.image_data.path), on_hot_class])

    return dataset_data_list, dataset_data

def get_image_uri(dataset_data, project):
    return os.path.join(file_action.get_static_dataset_root_path(project), dataset_data.dataset.name, dataset_data.image_data.path)

def get_unique_id():
    return int(time.time() * 1000)

def make_legacy_image_list(dataset_path):
    legacy_train_list = []
    legacy_val_list = []
    legacy_test_list = []
    legacy_image_list = []

    # read original train/val/test.txt
    with open(os.path.join(dataset_path,"train.txt"),"r") as f:
        legacy_train_list = f.read().split("\n")
    with open(os.path.join(dataset_path,"val.txt"),"r") as f:
        legacy_val_list = f.read().split("\n")
    with open(os.path.join(dataset_path,"test.txt"),"r") as f:
        legacy_test_list = f.read().split("\n")

    # connect list
    legacy_image_list = legacy_train_list + legacy_val_list + legacy_test_list
    random.shuffle(legacy_image_list)

    # make image_list
    with open(os.path.join(dataset_path, 'image_list.txt'), 'w') as f: 
        f.write('\n'.join(legacy_image_list))

def make_custom_test_list(dataset_path, class_list):
    # params
    test = []

    for class_number,class_name in enumerate(class_list):
        class_path = os.path.join(dataset_path, class_name)
        image_names = os.listdir(class_path)
        image_list = []
        
        for i in image_names:
            image_path = os.path.join(class_path,i)
            test.append(f'"{image_path}" {class_number}')

    else:
        # make text file
        with open(os.path.join(dataset_path, 'test.txt'), 'w') as f: 
            f.write('\n'.join(test))

def load_class_list(dataset):
    return list(ClassData.objects.filter(dataset=dataset).order_by('class_id').values_list("name", flat=True))

def get_pred_data(test_result, project_id):
    pred_records = test_result.pred_set.all()

    pred = []
    pred_prob = []
    labels = []
    label_names = []
    pred_names = []
    image_data = []
    max_pred_prob = []

    class_data_list = ClassData.objects.filter(dataset=pred_records[0].image_data.dataset).order_by('class_id')
    
    for pred_record in pred_records:
        pred.append(pred_record.pred)
        pred_names.append(class_data_list[pred_record.pred].name)
        labels.append(pred_record.image_data.class_data.class_id)
        label_names.append(pred_record.image_data.class_data.name)
        image_data.append(file_action.get_image_uri(project_id, pred_record.image_data.dataset, pred_record.image_data.path))

        pred_prob_records = pred_record.predprobe_set.all()
        pred_prob.append(np.zeros(len(pred_prob_records), np.float32))

        for i, pred_prob_record in enumerate(pred_prob_records):
            pred_prob[-1][i] = pred_prob_record.value
        max_pred_prob.append(pred_prob[-1].max())

    pred = np.array(pred, np.int)
    pred_prob = np.array(pred_prob)
    labels = np.array(labels, np.int)
    results = (pred == labels)
    max_pred_prob = np.array(max_pred_prob, np.float32)

    return pred, pred_prob, max_pred_prob, pred_names, labels, label_names, results, image_data


def get_pred_info(test_result, pred, labels):
    pred_records = test_result.pred_set.all()

    class_data_list = ClassData.objects.filter(dataset=pred_records[0].image_data.dataset).order_by('class_id')
    class_names = [class_data.name for class_data in class_data_list]

    # create confusion matrix
    n_classes = len(class_data_list)
    confusion_matrix = np.zeros((n_classes, n_classes), np.int)
    for label, pred in zip(labels, pred):
        confusion_matrix[label][pred] += 1

    accuracy = []
    precision = []
    recall = []
    fscore = []

    for i in range(n_classes):
        # precision  TP / (TP + FP)
        s = confusion_matrix[:, i].sum()
        if s == 0:
            precision.append(0.0)
        else:
            precision.append(confusion_matrix[i][i] / s)

        # recall  TP / (TP + FN)
        s = confusion_matrix[i, :].sum()
        
        if s == 0:
            recall.append(0.0)
        else:
            recall.append(confusion_matrix[i][i] / s)

        # F-score
        s = precision[-1] + recall[-1]
        if s == 0:
            fscore.append(0.0)
        else:
            fscore.append((2 * precision[-1] * recall[-1]) / s)
        
    return confusion_matrix, precision, recall, fscore