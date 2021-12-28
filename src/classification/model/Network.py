# coding: utf-8

import cv2, os, sys
from PIL import Image
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Input

from .Models import GoogLeNetModel
from .Models import VGG16Model
from .Models import InceptionV3Model
from .Models import MobileNetModel
from .Models import ResNet50Model

from . import const
from . import DA
from . import DA_setting

from main.log import get_logger

logger = get_logger(__name__)

class BaseNetwork(object):
    def __init__(self, **params):
        self.channel = params['channel'] if 'channel' in params else 3
        self.classes = params['classes'] if 'classes' in params else 1
        self.network = params['network']
        self.input_size = params['input_size'] if 'input_size' in params else None
        self.mean_image = params['mean_image'] if 'mean_image' in params else None
        self.image_type = params['image_type'] if 'image_type' in params else None
        self.xn = None
        self.yn = None
        self.val_xn = None
        self.val_yn = None
        self.pred_xn = None
        self.pred_yn = None
        
    def generate_train_data(self, train_list, da, batch_size):
        # count = 0
        while True:
            for data in train_list:
                # count += 1

                # get image(np.ndarray)
                image = self._get_image_array(data[0],
                                              resize=self.input_size,
                                              dtype=np.uint8,
                                              normalization=False)

                # for galleria
                y = data[1]
                
                # Data augmentation
                if len(data) < 3:
                    da_info = [[DA.NON_DA], [DA.NON_DA]]
                else:
                    da_info = data[2]

                da_im = da.get_image(image, da_info[0], da_info[1])

                # test code
                #savedir = ""
                #savename = "test_{}.jpg".format(count)
                #savepath = os.path.join(savedir,savename)
                #save_arr = Image.fromarray(np.uint8(da_im))
                #save_arr.save(savepath)

                da_im = da_im[np.newaxis,:,:,:]
                da_im = da_im.astype(np.float32)
                da_im /= 255

                if self.xn is None:
                    self.xn = da_im
                    self.yn = y
                else:
                    self.xn = np.vstack((self.xn, da_im))
                    self.yn = np.vstack((self.yn, y))

                if len(self.xn) == batch_size:
                    input_xn = self.xn
                    input_yn = self.yn
                    self.xn = None
                    self.yn = None
                    if self.network == const.GOOGLE_NET:
                        yield(input_xn,
                        {'loss1': input_yn,
                        'loss2': input_yn,
                        'loss3': input_yn})
                    else:
                        yield(input_xn, input_yn)

    def generate_val_data(self, val_list, da, batch_size):
        # count = 0
        while True:
            for data in val_list:
                # count += 1

                # get image(np.ndarray)
                image = self._get_image_array(data[0],
                                              resize=self.input_size,
                                              dtype=np.uint8,
                                              normalization=False)
                
                # for galleria
                y = data[1]
                
                # Data augmentation
                if len(data) < 3:
                    da_info = [[DA.NON_DA], [DA.NON_DA]]
                else:
                    da_info = data[2]

                da_im = da.get_image(image, da_info[0], da_info[1])

                # test code
                #savedir = ""
                #savename = "val_{}.jpg".format(count)
                #savepath = os.path.join(savedir,savename)
                #save_arr = Image.fromarray(np.uint8(da_im))
                #save_arr.save(savepath)

                da_im = da_im[np.newaxis,:,:,:]
                da_im = da_im.astype(np.float32)
                da_im /= 255

                if self.val_xn is None:
                    self.val_xn = da_im
                    self.val_yn = y
                else:
                    self.val_xn = np.vstack((self.val_xn, da_im))
                    self.val_yn = np.vstack((self.val_yn, y))

                if len(self.val_xn) == batch_size:
                    input_xn = self.val_xn
                    input_yn = self.val_yn
                    self.val_xn = None
                    self.val_yn = None
                    if self.network == const.GOOGLE_NET:
                        yield(input_xn,
                        {'loss1': input_yn,
                        'loss2': input_yn,
                        'loss3': input_yn})
                    else:
                        yield(input_xn, input_yn)

    def generate_predict_data(self, test_list, batch_size):
        while True:
            for data in test_list:
                image = self._get_image_array(data[0], #train_path,
                                              resize=self.input_size,
                                              dtype=np.uint8,
                                              normalization=False)
                
                image = image[np.newaxis,:,:,:]
                image = image.astype(np.float32)
                image /= 255
                
                if self.pred_xn is None:
                    self.pred_xn = image
                else:
                    self.pred_xn = np.vstack((self.pred_xn, image))

                if len(self.pred_xn) == batch_size:
                    input_xn = self.pred_xn
                    self.pred_xn = None
                    yield(input_xn)
            
    def _get_image_array(self, path, **params):
        
        dtype = params['dtype'] if 'dtype' in params else np.float32
        resize = params['resize'] if 'resize' in params else None
        normalization = params['normalization'] if 'normalization' in params else False    

        if self.channel == 1:
            #img = Image.open(path).convert('L')
            img = Image.open(path).convert('RGB')
        elif self.channel == 3:    
            img = Image.open(path).convert('RGB')
        else:
            img = Image.open(path).convert('RGB')

        im_arr = np.asarray(img)
        
        if resize is not None:
            im_arr = cv2.resize(im_arr, tuple(resize), interpolation=cv2.INTER_CUBIC)

        # 8bit image convert [w,h,1]
        # 32 bit image keep [w,h,3]
        if im_arr.ndim == 2:
            im_arr = im_arr[:,:,np.newaxis]
        # maybe RGBA type image protection
        if im_arr.ndim == 4:
            im_arr = im_arr[:,:,:3]
        im_arr = im_arr.astype(dtype)
        
        # use mean image
        if self.mean_image is not None:
            mean = Image.open(self.mean_image).convert('RGB')
            mean_arr = np.asarray(mean)
            im_arr -= mean_arr

        if normalization == True:
            im_arr /= 255
        
        return im_arr

    '''
    def _resize_array(self, image):
        if image.shape[0] != self.input_size[0] or image.shape[1] != self.input_size[1]:
            if image.dtype == np.float32 or image.dtype == np.float64:
                if K.image_dim_ordering() == 'th':
                    image = image[0,:,:]
                else:
                    image = image[:,:,0]

            im = Image.fromarray(image)
            im = im.resize(self.input_size, resample=Image.BICUBIC)
            image = np.asarray(im)
            if K.image_dim_ordering() == 'th':
                image = image[np.newaxis,:,:]
            else:
                image = image[:,:,np.newaxis]
        return image
    '''
        
class Network(BaseNetwork):
    def __init__(self, **params):

        super(Network,self).__init__(**params)

        input_tensor = Input(shape=(self.input_size[0], self.input_size[1], self.channel))
        # input_tensor = Input(shape=(self.input_size[0], self.input_size[1], 3))
        self.model = None

        logger.debug(self.network)
        if self.network == const.GOOGLE_NET:
            # self.model = InceptionV3Model(self.classes,input_tensor).model
            # self.model = GoogLeNetModel(self.classes, None, self.channel, self.input_size).model
            self.model = GoogLeNetModel(self.classes, None, 3, self.input_size).model
            
        elif self.network == const.VGG16:            
            self.model = VGG16Model(self.classes,input_tensor).model
            
        elif self.network == const.MOBILE_NET:
            self.model = MobileNetModel(self.classes,input_tensor).model
            
        elif self.network == const.RESNET50:
            self.model = ResNet50Model(self.classes,input_tensor).model
            
        # self.model.summary()

    def train(self, train_data, val_data, **params):
        epochs = params['epochs'] if 'epochs' in params else 1
        callbacks = params['callbacks'] if 'callbacks' in params else None
        batch = params['batch'] if 'batch' in params else 1
        val_batch = params['val_batch'] if 'val_batch' in params else 1
        da_params = params['data_augmentation'] if 'data_augmentation' in params else None

        da= DA_setting.run(da_params)
        da_instance = DA.DataAugmentation(da)
        train_data = da_instance.create_data_list(train_data)
        val_data = da_instance.create_data_list(val_data)

        train_data_batch_num = len(train_data) // batch
        if train_data_batch_num < 1:
            logger.debug('train_data_batch_num < 1')
            sys.exit(1)

        if val_data is not None:
            val_data_batch_num = len(val_data) // val_batch
            logger.debug(val_data_batch_num)
            if val_data_batch_num < 1:
                logger.debug('val_data_batch_num < 1')
                sys.exit(1)
                
            self.model.fit(
                self.generate_train_data(train_data, da_instance, batch),
                steps_per_epoch=train_data_batch_num,
                epochs=epochs,
                validation_data=self.generate_val_data(val_data, da_instance, val_batch),
                validation_steps=val_data_batch_num,
                callbacks=callbacks,
                verbose=1)
        else:
            self.model.fit(
            self.generate_train_data(train_data, da_instance, batch),
            steps_per_epoch=train_data_batch_num,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1)

    def save(self, path):
        self.model.save(path)

    def predict(self, data_list, **params):
        batch = params['batch'] if 'batch' in params else 1

        return self.model.predict_generator(
                self.generate_predict_data(data_list, batch),#, da_instance),
                steps=len(data_list) // batch,
                verbose=1)
 
