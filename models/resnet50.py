import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K

"""
Code taken from: 
https://github.com/fchollet/deep-learning-models
"""

IMAGE_ORDERING = 'channels_last'
# download links
if IMAGE_ORDERING == 'channels_first':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
elif IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


# utility functions: one-sided padding
def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x


# utility functions: identity block
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if IMAGE_ORDERING == 'channels_last': bn_axis = 3
    else: bn_axis = 1
    # naming
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # sub-block1
    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # sub-block2
    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING ,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # sub-block3
    x = Conv2D(filters3 , (1, 1), data_format=IMAGE_ORDERING , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    # output activation
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


# utility functions: conv block
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters   
    if IMAGE_ORDERING == 'channels_last': bn_axis = 3
    else: bn_axis = 1
    # naming 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # sub-block1
    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # sub-block2
    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING  , padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # sub-block3
    x = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    # shortcut convs
    shortcut = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # output activation
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def get_resnet50_encoder(input_height=224, input_width=224, channels=3,
                         pretrained='imagenet', include_top=True, weights='imagenet',
                         input_tensor=None, input_shape=None, pooling=None, classes=1000):

    #assert input_height%32 == 0
    #assert input_width%32 == 0
    if IMAGE_ORDERING == 'channels_first':
        bn_axis = 1
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(input_height, input_width, channels))
    # sub-block1
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2), name='conv1')(x)
    f1 = x
    # sub-block2
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , data_format=IMAGE_ORDERING , strides=(2, 2))(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x )
    # sub-block3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x 
    # sub-block4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x 
    # sub-block5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x 
    x = AveragePooling2D((7, 7) , data_format=IMAGE_ORDERING , name='avg_pool')(x)
    # f6 = x 
    if pretrained == 'imagenet':
        weights_path = keras.utils.get_file( pretrained_url.split("/")[-1] , pretrained_url  )
        Model(  img_input , x  ).load_weights(weights_path)
    return img_input, [f1 , f2 , f3 , f4 , f5]


def vanilla_encoder(input_height=224,  input_width=224, channels=3):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height,input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height,input_width, channels))
    x = img_input
    levels = []
    x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))( x )
    x = (Conv2D(filter_size, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))( x )
    x = (BatchNormalization())( x )
    x = (Activation('relu'))( x )
    x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING  ))( x )
    levels.append( x )

    x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))( x )
    x = (Conv2D(128, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))( x )
    x = (BatchNormalization())( x )
    x = (Activation('relu'))( x )
    x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING  ))( x )
    levels.append( x )

    for _ in range(3):
        x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))(x)
        x = (Conv2D(256, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING))(x)
        levels.append( x )
    return img_input, levels


def get_resnet_encoder(input_height=224, input_width=224, channels=3):
    img_input = Input(shape=(input_height, input_width, channels)) ; print (img_input)
    # sub-block1
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(32, (5, 5), data_format=IMAGE_ORDERING, strides=(2, 2), name='conv1')(x)
    f1 = x  ; print (f1)
    # sub-block2
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , data_format=IMAGE_ORDERING , strides=(2, 2))(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')
    f2 = one_side_pad(x ) ; print (f2)
    # sub-block3
    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')
    f3 = x  ; print (f3)
    # return
    return img_input, [f1 , f2 , f3]


