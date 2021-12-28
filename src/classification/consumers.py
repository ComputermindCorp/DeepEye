# django libs
from django.http import FileResponse
from django.core.serializers import serialize 
from django.utils import translation
from django.utils.translation import gettext

# deepeye setting & models & form
from main.models import Project
from .models import ClassificationModel, Dataset, Result, Weight
from .models import TestResult, Pred, PredProbe, TrainLog
from .model.train import main as classification_train
from .model.test import main as classification_test
from .model import DA
from .dataset_util import *

# common libs
from channels.generic.websocket import WebsocketConsumer
import glob
import json
import logging
import numpy as np
import os
import shutil
import sys
from threading import Thread
import time
import urllib

from main.project_type import ProjectType
from main import file_action

from main.log import get_logger

logger = get_logger(__name__)

class Classification(WebsocketConsumer):

    def connect(self):
        self.accept()
        
    def disconnect(self, close_code):
        pass

    def websocket_receive(self, data):
        logger.debug("[websocket_receive] data: {}".format(data))
        data = json.loads(data['text'])
        self.status = data['status']
        logger.info(f"Data received from frontend with status of '{self.status}'")
        if 'project_type' in data:
            project_type = data['project_type']

        if self.status == "lang-setting":
            translation.activate(data["user-lang"])

        # starting training
        elif self.status == 'train':
            thread = Thread(target = self.train, args = (data,))
            thread.start()
            # self.train(data)
        elif self.status == 'stop':
            pass

        elif self.status == 'training-ended':
            pass
        # testing( upload_data / self_testdata / saved_dataset )
        elif self.status == 'test':
            self.predict(data)
        
        # memo update
        """
        elif self.status == 'memo_update': 
            if data["target_type"] == "dataset":
                dataset = Dataset.objects.get(project=self.selected_project, name=data["selectedDatasetId"])
                dataset.memo = data["memo"]
                dataset.save()
            elif data["target_type"] == "model":
                model = ClassificationModel.objects.get(project=self.selected_project, name=data["selectedModelId"])
                model.memo = data["memo"]
                model.save()
            logger.debug("memo update finish")
        """
    def save_trainlog(self, data):
        training_model = ClassificationModel.objects.get(id=data['model_id'])
        train_log_record = TrainLog(
            epoch = data['epoch'],
            train_loss = data['train_loss'],
            train_acc = data['train_acc'],
            val_loss = data['val_loss'],
            val_acc = data['val_acc'],
            model = training_model,
        )
        train_log_record.save()

    def train(self, data):
        self.train_log = {}

        # get Dataset param from DB
        training_model = ClassificationModel.objects.get(id=data['model_id'])
        project = Project.objects.get(name=data['project_name'])
        dataset = Dataset.objects.get(name=data['dataset_name'],project=project)
        base_dataset_path = dataset.dataset_path
        default_test_ratio = dataset.default_test_ratio
        default_val_ratio = dataset.default_val_ratio

        # get training param from from
        model_name = data['model_name']
        architecture = data['architecture'].lower()
        epochs = int(data['epoch'])
        batch_size = int(data['batch'])
        learning_rate = float(data['learning_rate'])
        optimizer = data['optimizer'].lower()
    
        fine_tuning = data['fine_tuning'] 
        use_default_ratio = data['use_default_ratio']
        val_ratio = int(data['val_ratio'])
        test_ratio = int(data['test_ratio'])
        memo = data['memo']

        weights_path = data['weights_path']
        weights_file_path = data['weights_file_path']

        image_list_unique_id = data['image_list_unique_id']
        logger.debug(f"image_list_unique_id: {image_list_unique_id}")

        # make path & dir
        model_root = file_action.get_model_path_by_model_name(model_name, project)
        weights_path = file_action.get_weights_directory_by_model_name(model_name, project)
        dataset_path = os.path.join(model_root, "dataset")
        if os.path.exists(weights_path):
            shutil.rmtree(weights_path)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(weights_path, exist_ok=True)
        os.makedirs(dataset_path, exist_ok=True)

        # copy files
        class_list = load_class_list(dataset)

        num_classes = len(class_list)

        # make finetuning waight path
        if fine_tuning:
            transfer_path = training_model.baseweight_set.get().weight.path
        else:
            transfer_path = 'none'

        # get Augmentation flags
        augmentation_flags = {
            'horizontal_flip': data["horizontal_flip"],
            'vertical_flip': data["vertical_flip"],
            'rotate_30': data["rotate_30"],
            'rotate_45': data["rotate_45"],
            'rotate_90': data["rotate_90"],
            'gaussian_noise': data["gaussian_noise"],
            'blur': data["blur"],
            'contrast': data["contrast"]
        }
        image_type = data['image_type'].lower()
            
        if use_default_ratio:
            train_list, _ = get_dataset_list(project, dataset, DatasetDataType.Train)
            val_list, _ = get_dataset_list(project, dataset, DatasetDataType.Validation)
        else:
            train_list, _ = get_dataset_list(project, dataset, DatasetDataType.Train, image_list_unique_id)
            val_list, _ = get_dataset_list(project, dataset, DatasetDataType.Validation, image_list_unique_id)

        # Running training scripts
        try:
            classification_train(self,
                                data['model_id'],
                                model_root,
                                num_classes,
                                image_type,
                                train_list,
                                val_list,
                                augmentation_flags,
                                architecture,
                                epochs,
                                batch_size,
                                optimizer,
                                learning_rate,
                                transfer_path,
                                weights_path,
                                weights_file_path,
                                int(data["n_iter"]),
                                self.save_trainlog)
    
        except Exception as e:
            logger.debug(e)
            logger.debug('The program is exiting...')
            trans_message = gettext('training failed please check terminal')
            self.send(text_data=json.dumps({'status': 'reload',
                                            'message_type':'error',
                                            'message':trans_message}))

        finally:
            logger.debug("Saving Model to dataset")
            logger.debug(f"epoch: {self.train_log.get('epoch', '---')}")
            logger.debug(f"status: {self.train_log.get('status', '---')}")
            logger.debug(f"train_loss: {self.train_log.get('train_loss', '---')}")
            logger.debug(f"train_acc: {self.train_log.get('train_acc', '---')}")
            logger.debug(f"val_loss: {self.train_log.get('val_loss', '---')}")
            logger.debug(f"val_acc: {self.train_log.get('val_acc', '---')}")
            logger.debug(f"best_train_loss: {self.train_log.get('best_train_loss', '---')}")
            logger.debug(f"best_val_loss: {self.train_log.get('best_val_loss', '---')}")
            logger.debug(f"best_train_epoch: {self.train_log.get('best_train_epoch', '---')}")
            logger.debug(f"best_val_epoch: {self.train_log.get('best_val_epoch', '---')}")

            try:
                training_model.epochs_runned = self.train_log['epoch']
                training_model.train_status = self.train_log['status']
                training_model.train_loss = self.train_log['train_loss']
                training_model.train_acc = self.train_log['train_acc']
                training_model.val_loss = self.train_log['val_loss']
                training_model.val_acc = self.train_log['val_acc']
                training_model.best_train_loss = self.train_log['best_train_loss']
                training_model.best_val_loss = self.train_log['best_val_loss']
                training_model.best_train_epoch = self.train_log['best_train_epoch']
                training_model.best_val_epoch = self.train_log['best_val_epoch']
                                                                                        
                training_model.save()
            except:
                logger.info("fail: training model save")

            if not use_default_ratio:
                new_dataset_data = DatasetData.objects.filter(unique_id=image_list_unique_id)
                for data in new_dataset_data:
                    data.model = training_model
                    data.save()

            trans_message = gettext('training : {} training ended')

            cancel = self.train_log.get('cancel', '')
            if cancel == '':
                training_model.delete()               
            
            self.send(text_data=json.dumps({'status': 'reload',
                                            'message_type':'success',
                                            'message': trans_message.format(model_name),
                                            'cancel': cancel,
                                            'project_id': project.id}))
            sys.exit(0)

    def predict(self, data):
        # get common params from form
        project = Project.objects.get(pk=data["project_id"])
        project_type = ProjectType(project.project_type)
        model = ClassificationModel.objects.get(id=data["model_id"], project=project)
        weight = Weight.objects.get(model=model)

        predict_type = data['predict_type'] # self_dataset / save_dataset / upload_dataset
        if predict_type == "self_dataset":
            train_flag = data['train_flag']
            val_flag = data["val_flag"]
            test_flag = data["test_flag"]
        elif predict_type == "save_dataset":
            dataset = Dataset.objects.get(pk=data['database_id'])
            train_flag = data['train_flag']
            val_flag = data["val_flag"]
            test_flag = data["test_flag"]
        elif predict_type == "upload_dataset":
            train_flag = None
            val_flag = None
            test_flag = None

        # get model params from DB
        architecture = model.architecture_type
        num_classes = model.dataset.classes
        image_type = model.image_type

        model_root = file_action.get_model_path(model)
        training_class_list = load_class_list(model.dataset)

        #
        if predict_type == "self_dataset":
            dataset = model.dataset

            dataset_data_types = []
            if train_flag:
                dataset_data_types.append(DatasetDataType.Train)
            if val_flag:
                dataset_data_types.append(DatasetDataType.Validation)
            if test_flag:
                dataset_data_types.append(DatasetDataType.Test)

            predict_list, dataset_data_list = get_dataset_list(project, dataset, dataset_data_types)

        elif predict_type == "save_dataset":
            dataset_data_types = []
            if train_flag:
                dataset_data_types.append(DatasetDataType.Train)
            if val_flag:
                dataset_data_types.append(DatasetDataType.Validation)
            if test_flag:
                dataset_data_types.append(DatasetDataType.Test)

            predict_list, dataset_data_list = get_dataset_list(project, dataset, dataset_data_types)
        else:
            pass

        # run predict
        logger.debug("model.train_status: {}".format(model.train_status))
        if model.train_status == 'finished' or model.train_status == 'stopped':
            try:
                logger.debug(architecture)
                self.result = Result
                preds, pred_probs, labels = classification_test(
                                    self,
                                    model_root,
                                    project,
                                    model,
                                    num_classes,
                                    image_type,
                                    architecture,
                                    predict_list,
                                    training_class_list,
                                    weight.path)
                # delete test result database
                all_test_result = TestResult.objects.all()
                all_test_result.delete()

                # create database
                new_test_result = TestResult(model=model)
                new_test_result.save()
                for pred, pred_prob, label, dataset_data in zip(preds, pred_probs, labels, dataset_data_list):
                    new_pred = Pred(
                        test_result=new_test_result,
                        pred=pred,
                        model=model,                  
                        image_data=dataset_data.image_data    
                    )
                    new_pred.save()
                    
                    for p in pred_prob:
                        new_pred_prob = PredProbe(pred=new_pred, value=p)
                        new_pred_prob.save()

                self.send(text_data=json.dumps({
                    'status': 'test-complete',
                    'dataset_id': dataset.id,
                    'test_result_id': new_test_result.id,
                    }))

            except Exception as e:
                logger.debug('Testing exiting on error...')
                logger.debug(e)
                self.send(text_data=json.dumps({'status': 'error',
                                                'text': e}))

            finally:
                if predict_type == "self_dataset":
                    pass
                elif predict_type == "save_dataset":
                    pass
                elif predict_type == "upload_dataset":
                    logger.debug("Deleting upload files")
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        else:
            trans_message =_('Chosen model training not completed')
            self.send(text_data=json.dumps({'status': 'error',
                                            'text': trans_message}))
