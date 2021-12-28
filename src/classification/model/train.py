# DeepLerning libs
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard
)
from . import util
from . import const
from . import Network
from . import DA

# common libs
import csv
import glob
import json
import numpy as np
import os
import random
import shutil
import sys

from main.log import get_logger

logger = get_logger(__name__)

def main(socket,
        model_id:int,
        model_root:str,
        num_classes:int,
        image_type:str, 
        train_data:list,
        val_data:list, 
        augmentation:dict,
        architecture:str, 
        epochs:int,
        batch_size:int, 
        optimizer_str:str,
        learning_rate:float,
        transfer_path:str,
        weights_path:str,
        weights_file_path:str,
        n_iter:int,
        log_callback=None):

    socket.send(text_data=json.dumps({'status': 'process-update',
                                      'process':"preprocess"}))

    # GPU setting
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    logger.info(f"Model Selected {architecture}")

    ## debug
    logger.debug("train data: {}".format(train_data))
    logger.debug("val data: {}".format(val_data))
    ##

    lr_schedule = learning_rate
    if optimizer_str == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=lr_schedule)
    elif optimizer_str == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr=lr_schedule)
    elif optimizer_str == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(lr=lr_schedule)
    elif optimizer_str == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(lr=lr_schedule)
    elif optimizer_str == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=lr_schedule)
    elif optimizer_str == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(lr=lr_schedule)
    elif optimizer_str == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(lr=lr_schedule)

    # Setting callbacks
    if architecture == 'googlenet':
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                #logger.debug(logs)
                socket.send(text_data=json.dumps({'status': 'training',
                                                  'batch': batch+1,
                                                  'n_iter': n_iter,
                                                  }))
            def on_epoch_end(self, epoch, logs=None):
                #logger.debug(logs)
                logger.debug(socket.status)
                if epoch == 0:
                    self.best_train_loss = logs.get('loss')
                    self.best_val_loss = logs.get('val_loss')
                    self.best_train_epoch = 1
                    self.best_val_epoch = 1
                else:
                    if self.best_train_loss >= logs.get('loss'):
                        self.best_train_loss = logs.get('loss')
                        self.best_train_epoch = epoch + 1
                    if self.best_val_loss >= logs.get('val_loss'):
                        self.best_val_loss = logs.get('val_loss')
                        self.best_val_epoch = epoch + 1

                if log_callback is not None:
                    data = {
                        'epoch': epoch + 1,
                        'train_loss': logs.get('loss'),
                        'train_acc': logs.get('loss3_accuracy'),
                        'val_loss': logs.get('val_loss'),
                        'val_acc': logs.get('val_loss3_accuracy'),
                        'model_id': model_id
                    }
                    log_callback(data)

                if socket.status == 'stop':
                    logger.info("Training stopped from frontend")
                    logger.debug("set train_log 'stopped'")
                    socket.train_log = {'status':'stopped',
                                        'epoch':epoch + 1,
                                        'train_loss': logs.get('loss'),
                                        'train_acc': logs.get('loss3_accuracy'),
                                        'val_loss': logs.get('val_loss'),
                                        'val_acc': logs.get('val_loss3_accuracy'),
                                        'best_train_loss': self.best_train_loss,
                                        'best_val_loss': self.best_val_loss,
                                        'best_train_epoch': self.best_train_epoch,
                                        'best_val_epoch': self.best_val_epoch,
                                        'cancel': True }
                    socket.send(text_data=json.dumps({'status': 'training-stopped'}))
                    # stop save
                    net.save(weights_file_path)
                    # suspend
                    logger.debug("train process exit")
                    sys.exit(0)
                socket.send(text_data=json.dumps({'status': 'training',
                                                  'epoch':epoch + 1,
                                                  'epochs': epochs,
                                                  'train_loss': logs.get('loss'),
                                                  'train_acc': logs.get('loss3_accuracy'), 
                                                  'val_loss': logs.get('val_loss'),
                                                  'val_acc': logs.get('val_loss3_accuracy'),
                                                  'best_train_loss': self.best_train_loss,
                                                  'best_val_loss': self.best_val_loss,
                                                  'best_train_epoch': self.best_train_epoch,
                                                  'best_val_epoch': self.best_val_epoch,
                                                  }))

                logger.debug("set train_log 'finished'")
                socket.train_log = {'status': 'finished',
                                    'epoch':epoch + 1, 
                                    'train_loss': logs.get('loss'),
                                    'train_acc': logs.get('loss3_accuracy'), 
                                    'val_loss': logs.get('val_loss'),
                                    'val_acc': logs.get('val_loss3_accuracy'),
                                    'best_train_loss': self.best_train_loss,
                                    'best_val_loss': self.best_val_loss,
                                    'best_train_epoch': self.best_train_epoch,
                                    'best_val_epoch': self.best_val_epoch,
                                    'cancel': False
                                    }
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
            #EarlyStopping(monitor='loss', patience=3),
            CSVLogger(os.path.join(weights_path, 'training_log.csv')),
            ModelCheckpoint(filepath=os.path.join(weights_path, 'checkpoints', architecture+'-training{epoch:04d}.ckpt'),
                            save_weights_only=True, 
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            verbose=1),
        ]
    else:
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                socket.send(text_data=json.dumps({'status': 'training',
                                                  'batch':batch+1,
                                                  'n_iter': n_iter,
                                                  }))

            def on_epoch_end(self, epoch, logs=None):
                logger.debug(socket.status)
                if epoch == 0:
                    self.best_train_loss = logs.get('loss')
                    self.best_val_loss = logs.get('val_loss')
                    self.best_train_epoch = 1
                    self.best_val_epoch = 1
                else:
                    if self.best_train_loss >= logs.get('loss'):
                        self.best_train_loss = logs.get('loss')
                        self.best_train_epoch = epoch + 1
                    if self.best_val_loss >= logs.get('val_loss'):
                        self.best_val_loss = logs.get('val_loss')
                        self.best_val_epoch = epoch + 1

                if log_callback is not None:
                    data = {
                        'epoch': epoch + 1,
                        'train_loss': logs.get('loss'),
                        'train_acc': logs.get('accuracy'),
                        'val_loss': logs.get('val_loss'),
                        'val_acc': logs.get('val_accuracy'),
                        'model_id': model_id
                    }
                    log_callback(data)

                if socket.status == 'stop':
                    logger.info("Training stopped from frontend")
                    logger.debug("set train_log 'stopped'")
                    socket.train_log = {'status':'stopped',
                                        'epoch':epoch+1,
                                        'train_loss': logs.get('loss'),
                                        'train_acc': logs.get('accuracy'), 
                                        'val_loss': logs.get('val_loss'),
                                        'val_acc': logs.get('val_accuracy'),
                                        'best_train_loss': self.best_train_loss,
                                        'best_val_loss': self.best_val_loss,
                                        'best_train_epoch': self.best_train_epoch,
                                        'best_val_epoch': self.best_val_epoch,
                                        'cancel': True}
                    logger.debug(f"socket.train_log: {socket.train_log}")

                    socket.send(text_data=json.dumps({'status': 'training-stopped'}))
                    net.save(weights_file_path)
                    # suspend
                    sys.exit(0)
                socket.send(text_data=json.dumps({'status': 'training',
                                                  'epoch':epoch+1,
                                                  'epochs': epochs,
                                                  'train_loss': logs.get('loss'),
                                                  'train_acc': logs.get('accuracy'), 
                                                  'val_loss': logs.get('val_loss'),
                                                  'val_acc': logs.get('val_accuracy'),
                                                  'best_train_loss': self.best_train_loss,
                                                  'best_val_loss': self.best_val_loss,
                                                  'best_train_epoch': self.best_train_epoch,
                                                  'best_val_epoch': self.best_val_epoch,
                                                  }))
                                                  
                logger.debug("set train_log 'finished'")
                socket.train_log = {'status': 'finished',
                                    'epoch':epoch+1,
                                    'train_loss': logs.get('loss'),
                                    'train_acc': logs.get('accuracy'), 
                                    'val_loss': logs.get('val_loss'),
                                    'val_acc': logs.get('val_accuracy'),
                                    'best_train_loss': self.best_train_loss,
                                    'best_val_loss': self.best_val_loss,
                                    'best_train_epoch': self.best_train_epoch,
                                    'best_val_epoch': self.best_val_epoch,
                                    'cancel': False}

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
            #EarlyStopping(monitor='loss', patience=3),
            CSVLogger(os.path.join(weights_path, 'training_log.csv')),
            ModelCheckpoint(filepath=os.path.join(weights_path, 'checkpoints', architecture+'-training{epoch:04d}.ckpt'),
                            save_weights_only=True, 
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            verbose=1),
        ]


    size = [224,224]
    net = Network.Network(channel=3, classes=num_classes, network=architecture, input_size=size,image_type=image_type)

    if architecture == 'googlenet':
        net.model.compile(
                loss={
                    'loss1': 'categorical_crossentropy',
                    'loss2': 'categorical_crossentropy',
                    'loss3': 'categorical_crossentropy',
                },
                loss_weights={
                    'loss1': 0.3,
                    'loss2': 0.3,
                    'loss3': 1.0,
                },
                optimizer=optimizer,
                metrics=['accuracy'])
    else:
        net.model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    if transfer_path == "none":
        pass
    else:
        try:
            logger.debug("transfer_path : {}".format(transfer_path))
            socket.send(text_data=json.dumps({'status': 'process-update',
                                              'process':"finetuning"}))
            net.model.load_weights(transfer_path)
            logger.debug("pretrain weight load")
        except Exception as e:
            logger.debug("pretrain weight load failed")
            logger.debug(e)
            socket.send(text_data=json.dumps({'status': 'error',
                                              'text': 'weight load failed'}))
            sys.exit(0)
    
    # data shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)

    socket.send(text_data=json.dumps({'status': 'process-update',
                                      'process':"training"}))

    # training start
    net.train(train_data, val_data,
              epochs=epochs,
              batch=batch_size,
              data_augmentation=augmentation,
              callbacks=[[CustomCallback()], callbacks])

    socket.send(text_data=json.dumps({'status': 'process-update',
                                      'process':"postprocess"}))

    net.save(weights_file_path)
    shutil.rmtree(os.path.join(weights_path, 'checkpoints'), ignore_errors=True)

    logger.info('Training Completed')
    socket.send(text_data=json.dumps({'status': 'training-ended'}))
    