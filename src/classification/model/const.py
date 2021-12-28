
################################################
# list file name define

# train data list file name
TRAIN_TXT = 'train.txt'

# validataion data list file name
VAL_TXT = 'val.txt'

# test data list file name
TEST_TXT = 'test.txt'

################################################



################################################
# output file/dir name define

# model file output dir name
MODEL_OUTPUT_DIR = 'Models'

# train log file name
TRAIN_LOG_FILE = 'log.csv'

# NN model name
MODEL_FILE_NAME = 'weights.{epoch:d}.hdf5'
MODEL_FILE_NAME_BESTONLY = 'weights.hdf5'

# predict result file name
PREDICT_RESULT_FILE = 'predict.csv'

################################################

################################################
# network name define

GOOGLE_NET = 'googlenet'
VGG16 = 'vgg16'
MOBILE_NET = 'mobilenet'
RESNET50 = 'resnet50'
################################################

# GradCAM target layer name

#GOOGLE_NET_GRAD_LAYER = 111
#VGG16_GRAD_LAYER = 15
#MOBILE_NET_GRAD_LAYER = 64
#RESNET50_GRAD_LAYER = 128

GOOGLE_NET_GRAD_LAYER = 'inc5b_out5'
VGG16_GRAD_LAYER = 'block5_conv3'
MOBILE_NET_GRAD_LAYER = 'conv_pw_13_relu'
RESNET50_GRAD_LAYER = 'activation_49'
