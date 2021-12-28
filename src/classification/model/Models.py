from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Activation, Dropout, Dense, concatenate, GlobalAveragePooling2D,add
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50

class Models(object):
    def __init__(self,classes, input_tensor):
        self.classes = classes
        self.input_tensor = input_tensor

# Not Use
class InceptionV3Model(Models):
    def __init__(self,classes, input_tensor):
        super(InceptionV3Model,self).__init__(classes, input_tensor)

        incpv3 = InceptionV3(weights='imagenet', include_top=False, input_tensor=self.input_tensor)

        for layer in incpv3.layers:
            layer.trainable = True

        x = incpv3.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.classes, activation='softmax')(x)
        self.model = Model(inputs=incpv3.input, outputs=predictions)

        for layer in self.model.layers:
            layer.trainable = True

# use
class GoogLeNetModel(Models):
    def __init__(self,classes, input_tensor,channel,size):
        super(GoogLeNetModel,self).__init__(classes, input_tensor)
        self.channel = channel
        self.input_size = size

        # if K.image_dim_ordering() == 'th':
        #     input = Input(shape=(self.channel, self.input_size[0],self.input_size[1]))
        # else:
        
        input = Input(shape=(self.input_size[0],self.input_size[1],self.channel))
       
        conv1 = ZeroPadding2D((3,3))(input)
        conv1 = Conv2D(64, (7,7), strides=(2,2), activation='relu', name='conv1')(conv1)
        conv1 = ZeroPadding2D((1,1))(conv1)
        conv1 = MaxPooling2D((3,3), strides=(2,2))(conv1)
        conv1 = BatchNormalization()(conv1)

        conv2 = Conv2D(64, (1,1), activation='relu', name='conv2_reduce')(conv1)
        conv2 = ZeroPadding2D((1,1))(conv2)
        conv2 = Conv2D(192, (3,3), strides=(1,1), activation='relu', name='conv2')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = ZeroPadding2D((1,1))(conv2)
        conv2 = MaxPooling2D((3,3), strides=(2,2))(conv2)

        inc3a = self.inception(conv2, 64, 96, 128, 16, 32, 32, 'inc3a')
        inc3b = self.inception(inc3a, 128, 128, 192, 32, 96, 64, 'inc3b')

        inc4a = ZeroPadding2D((1,1))(inc3b)
        inc4a = MaxPooling2D((3,3), strides=(2,2))(inc4a)
        inc4a = self.inception(inc4a, 192, 96, 208, 16, 48, 64, 'inc4a')

        loss1 = AveragePooling2D((5,5), strides=(3,3))(inc4a)
        loss1 = Conv2D(128, (1,1), activation='relu', name='loss1_conv1')(loss1)
        loss1 = Flatten()(loss1)
        loss1 = Dense(1024, activation='relu', name='loss1_dense1')(loss1)
        loss1 = Dense(classes, name='loss1_dense2')(loss1)
        loss1 = Activation('softmax', name='loss1')(loss1)

        inc4b = self.inception(inc4a, 160, 112, 224, 24, 64, 64, 'inc4b')
        inc4c = self.inception(inc4b, 128, 128, 256, 24, 64, 64, 'inc4c')
        inc4d = self.inception(inc4c, 112, 144, 288, 32, 64, 64, 'inc4d')

        loss2 = AveragePooling2D((5,5), strides=(3,3))(inc4d)
        loss2 = Conv2D(128, (1,1), activation='relu', name='loss2_conv1')(loss2)
        loss2 = Flatten()(loss2)
        loss2 = Dense(1024, activation='relu', name='loss2_dence1')(loss2)
        loss2 = Dense(classes, name='loss2_dence2')(loss2)
        loss2 = Activation('softmax', name='loss2')(loss2)

        inc4e = self.inception(inc4d, 256, 160, 320, 32, 128, 128, 'inc4e')
        inc4e = ZeroPadding2D((1,1))(inc4e)
        inc4e = MaxPooling2D((3,3), strides=(2,2))(inc4e)
        inc5a = self.inception(inc4e, 256, 160, 320, 32, 128, 128, 'inc5a')
        inc5b = self.inception(inc5a, 384, 192, 384, 48, 128, 128, 'inc5b')

        loss3 = AveragePooling2D((7,7), strides=(1,1))(inc5b)
        loss3 = Flatten()(loss3)
        loss3 = Dropout(0.4)(loss3)
        loss3 = Dense(classes, name='loss3_dence')(loss3)
        loss3 = Activation('softmax', name='loss3')(loss3)

        self.model = Model(inputs=input, outputs=[loss1, loss2, loss3])
        
    def inception(self, x, out1, proj3, out3, proj5, out5, proj_pool, base_name):
        _out1 = Conv2D(out1, (1, 1), activation='relu', name=base_name+'_out1')(x)

        _out3 = Conv2D(proj3, (1,1), activation='relu', name=base_name+'_proj3')(x)
        _out3 = ZeroPadding2D((1,1))(_out3)
        _out3 = Conv2D(out3, (3,3), name=base_name+'_out3')(_out3)

        _out5 = Conv2D(proj5, (1,1), activation='relu', name=base_name+'_proj5')(x)
        _out5 = ZeroPadding2D((2,2))(_out5)
        _out5 = Conv2D(out5, (5,5), name=base_name+'_out5')(_out5)

        _pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        _pool = Conv2D(proj_pool, (1,1), name=base_name+'_pool')(_pool)

        _concat = concatenate([_out1, _out3, _out5, _pool], axis=3)
        _concat = Activation('relu', name=base_name+'_concat')(_concat)

        return _concat
            
class VGG16Model(Models):
    def __init__(self,classes, input_tensor):
        super(VGG16Model,self).__init__(classes, input_tensor)

        vgg16 = VGG16(weights=None, include_top=True, classes=classes, input_tensor=self.input_tensor)
        for layer in vgg16.layers:
            layer.trainable = True

        predictions = vgg16.output
        self.model = Model(inputs=vgg16.input, outputs=predictions)

class MobileNetModel(Models):
    def __init__(self,classes, input_tensor):
        super(MobileNetModel,self).__init__(classes, input_tensor)

        mobilenet = MobileNet(weights=None, include_top=True, classes=classes, input_tensor=self.input_tensor)

        for layer in mobilenet.layers:
            layer.trainable = True

        predictions = mobilenet.output
        self.model = Model(inputs=mobilenet.input, outputs=predictions)

class ResNet50Model(Models):
    def __init__(self,classes, input_tensor):
        super(ResNet50Model,self).__init__(classes, input_tensor)
        
        resne50 = ResNet50(weights=None, include_top=True, classes=classes,input_tensor=self.input_tensor)
        for layer in resne50.layers:
            layer.trainable = True
        
        predictions = resne50.output
        
        self.model = Model(inputs=resne50.input, outputs=predictions)