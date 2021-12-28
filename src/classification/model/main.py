import os
import glob
import util as u
from argparse import ArgumentParser
import train
import predict

from main.log import get_logger

logger = get_logger(__name__)

class Params():
    def __init__(self):
        args = self.parser()

        self.is_train = args.train
        size = str(args.size).split(',')
        self.size = [int(size[0]),int(size[1])]
        self.classes = args.classes
        self.channel = args.ch
        self.data_root_dir = args.data_root_dir
        self.epoch = args.epoch
        self.snap_shot = args.snap_shot
        self.save_best_only = args.save_best_only
        self.batch_size = args.batch_size
        self.base_lr = args.base_lr
        self.optimizer = args.optimizer
        self.weights = args.weights
        self.result_dir = args.result_dir
        self.net = args.net
        self.data_augmentation = args.data_augmentation
        self.mean_image = args.mean_image
        
    def parser(self):
        argparser = ArgumentParser()

        # --size=100,100
        argparser.add_argument('--train',type=int,default=-1)
        argparser.add_argument('--size',type=str,default="224,224")
        argparser.add_argument('--classes',type=int,default=3)
        argparser.add_argument('--ch',type=int,default=3)
        argparser.add_argument('--data_root_dir',type=str,default='/home')
        argparser.add_argument('--epoch',type=int,default=100)
        argparser.add_argument('--snap_shot',type=int,default=1)
        argparser.add_argument('--save_best_only',type=str,default='False')
        argparser.add_argument('--batch_size',type=int,default=1)
        argparser.add_argument('--base_lr',type=float,default=0.001)
        argparser.add_argument('--optimizer',type=str,default='sgd')
        argparser.add_argument('--weights',type=str)
        argparser.add_argument('--result_dir',type=str,default='/home/results')
        argparser.add_argument('--net',type=str,default='googlenet')
        argparser.add_argument('--data_augmentation', type=int, default=0)
        argparser.add_argument('--mean_image',type=str)
        
        args = argparser.parse_args()
        args.save_best_only = (args.save_best_only.lower() == 'true')

        return args

if __name__ == '__main__':
    p = Params()
    
    if p.is_train == 1:
        logger.debug('Train !')
        train.Train(p)
        
    elif p.is_train == 0:
        logger.debug('test !')
        predict.Predict(p)
