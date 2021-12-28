from . import DA
from .  import DA_setting
import cv2
import numpy as np
from argparse import ArgumentParser
import sys

from main.log import get_logger

logger = get_logger(__name__)

class Params():
    def __init__(self):
        args = self.parser()

        self.h_flip = args.h_flip
        self.v_flip = args.v_flip
        self.rotate_90s = args.rotate_90s
        self.rotate_45s = args.rotate_45s
        self.rotate_30s = args.rotate_30s
        self.gaussian_noise = args.gaussian_noise
        self.blur = args.blur
        self.low_contrast = args.low_contrast
        self.hi_contrast = args.hi_contrast
        self.input_img = args.input_img
        self.output_img = args.output_img
        #self.data_augmentation = args.data_augmentation
        
    def parser(self):
        argparser = ArgumentParser()
        
        '''
        argparser.add_argument('--h_flip',type=str,default='False')
        argparser.add_argument('--v_flip',type=str,default='False')
        argparser.add_argument('--rotate_90s',type=str,default='False')
        argparser.add_argument('--rotate_45s',type=str,default='False')
        argparser.add_argument('--rotate_30s',type=str,default='False')
        argparser.add_argument('--gaussian_noise',type=str,default='False')
        argparser.add_argument('--blur',type=str,default='False')
        argparser.add_argument('--contrast',type=str,default='False')
        argparser.add_argument('--input_img',type=str)
        argparser.add_argument('--output_img',type=str)
        '''
        
        argparser.add_argument('--h_flip', action='store_true')
        argparser.add_argument('--v_flip', action='store_true')
        argparser.add_argument('--rotate_90s', action='store_true')
        argparser.add_argument('--rotate_45s', action='store_true')
        argparser.add_argument('--rotate_30s', action='store_true')
        argparser.add_argument('--gaussian_noise', action='store_true')
        argparser.add_argument('--blur', action='store_true')
        argparser.add_argument('--low_contrast', action='store_true')
        argparser.add_argument('--hi_contrast', action='store_true')
        argparser.add_argument('--input_img',type=str)
        argparser.add_argument('--output_img',type=str)
        
        args = argparser.parse_args()

        return args

def make_da(params):
    da = 0
    if params.h_flip == True:
        da += 0x0001
    if params.v_flip == True:
        da += 0x0002
    if params.rotate_90s == True:
        da += 0x0004
    if params.rotate_45s == True:
        da += 0x0008
    if params.rotate_30s == True:
        da += 0x0010
    if params.gaussian_noise == True:
        da += 0x0020
    if params.blur == True:
        da += 0x0040
    if (params.hi_contrast == True or params.low_contrast == True):
        da += 0x0080
    return da

def GenerateSample(params):
    logger.debug('h_flip {}'.format(params.h_flip))
    logger.debug('v_flip {}'.format(params.v_flip))
    logger.debug('rotate_90s {}'.format(params.rotate_90s))
    logger.debug('rotate_45s {}'.format(params.rotate_45s))
    logger.debug('rotate_30s {}'.format(params.rotate_30s))
    logger.debug('gaussian_noise {}'.format(params.gaussian_noise))
    logger.debug('blur {}'.format(params.blur))
    logger.debug('hi_contrast {}'.format(params.hi_contrast))
    logger.debug('low_contrast {}'.format(params.low_contrast))
    logger.debug('input_img {}'.format(params.input_img))
    logger.debug('output_img {}'.format(params.output_img))

    da = DA_setting.get_da_list(make_da(params))
    da_instance = DA.DataAugmentation(da)
    
    img = cv2.imread(params.input_img, -1)
    
    img = np.asarray(img)
    if img.ndim == 2:
            img = img[:,:,np.newaxis]
            logger.debug("new axis arrived!!!")
    if img.ndim == 4:
            logger.debug(img)
            img = img[:,:,:3]
            logger.debug(img)
            logger.debug("alpha channel detected")
            logger.debug(img.shape)
    img.astype(np.uint8)
        
    if params.low_contrast == True:
        img = da_instance.lo_contrast(img)
    if params.hi_contrast == True:
        img = da_instance.hi_contrast(img)
    if params.h_flip == True:
        img = img[:,::-1, :]
    if params.v_flip == True:
        img = img[::-1,:, :]
    if params.rotate_90s == True:
        img = da_instance.rotate(img, 90)
    if params.rotate_45s == True:
        img = da_instance.rotate(img, 45)
    if params.rotate_30s == True:
        img = da_instance.rotate(img, 30)
    if params.gaussian_noise == True:
        img = da_instance.gaussian_noise(img)
    if params.blur == True:
        img = da_instance.blur(img, 10)
    
    
    cv2.imwrite(params.output_img, img)
    
    logger.debug('output successful')

if __name__ == '__main__':
    p = Params()
    GenerateSample(p)

