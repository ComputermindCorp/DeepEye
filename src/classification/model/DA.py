import cv2
import scipy.ndimage.interpolation
import numpy as np
from . import DA_setting

from main.log import get_logger

logger = get_logger(__name__)

NON_DA = 0x0000
H_FLIP = 0x0001
V_FLIP = 0x0002
ROTATE = 0x0004
GAUSSIAN_NOISE = 0x0008
BLUR = 0x0010
CONTRAST = 0x0020

class DataAugmentation:
    def __init__(self, da_lists):
        self.base_da_list = da_lists[0]
        self.da_list = da_lists[1]

    def get_image(self, im, base_da, da):
        if im.ndim == 2:
            im = im[:,:,np.newaxis]

        #if im.dtype == np.float32:
        #    im = im * 255
        #    im = im.astype(np.uint8)

        if base_da[0] == NON_DA:
            base_im = im
        elif base_da[0] == GAUSSIAN_NOISE:
            base_im = self.gaussian_noise(im)
        elif base_da[0] == BLUR:
            base_im = self.blur(im)
        elif base_da[0] == CONTRAST:
            if(base_da[1] == 0):
                base_im = self.hi_contrast(im)
            elif(base_da[1] == 1):
                base_im = self.lo_contrast(im)

            else:
                raise ValueError("Invalid Data")
        else:
            raise ValueError("Invalid Data Augmentation Type")

        if base_im.ndim == 2:
            base_im = base_im[:,:,np.newaxis]

        if da[0] == NON_DA:
            da_im = base_im
        elif da[0] == H_FLIP:
            da_im = base_im[:,::-1, :]
        elif da[0] == V_FLIP:
            da_im = base_im[::-1,:, :]
        elif da[0] == ROTATE:
            da_im = self.rotate(base_im, da[1], reshape=False)
        else:
            raise ValueError("Invalid Data Augmentation Type")

        if da_im.ndim == 2:
            da_im = da_im[:,:,np.newaxis] 

        return da_im


    def rotate(self, im, deg, reshape=False):
        
        org_h, org_w, _ = im.shape
        
        im_tmp = scipy.ndimage.interpolation.rotate(im, deg, reshape=reshape)
        im_tmp =  cv2.resize(im_tmp, (org_h,org_w), interpolation=cv2.INTER_CUBIC)

        return im_tmp

    def gaussian_noise(self, im, sigma=15):
        gauss = np.random.normal(0, sigma, im.shape)
        gauss = gauss.reshape(im.shape)
        return im + gauss
    
    def blur(self, im, filter=3):
        return cv2.blur(im, (filter,filter))

    def hi_contrast(self, im, low=50, hi=205):
        diff = hi - low
        
        LUT = np.zeros(256, dtype=im.dtype)
        
        for i in range(low, hi):
            LUT[i] = 255 * (i-low) / diff
        LUT[hi:256] = 255
        logger.debug(LUT)
            
        return cv2.LUT(im, LUT)
        
    def lo_contrast(self, im, low=50, hi=205):
        diff = hi - low
        
        LUT = np.zeros(256, dtype=im.dtype)

        for i in range(256):
            LUT[i] = low + i * diff / 255
        logger.debug(LUT)
        return cv2.LUT(im, LUT)

    def create_data_list(self, data_list):
        ret_list = []

        for data in data_list:
            for base_da in self.base_da_list:
                for da in self.da_list:
                    ret_list.append(data + [[base_da, da]])

        return ret_list

    def _get_length(self, n_data):
        return n_data * len(self.base_da_list) * len(self.da_list)

    @classmethod
    def get_length(cls, n_data, da_params):
        da = DA_setting.run(da_params)
        da_instance = cls(da)
        return da_instance._get_length(n_data)