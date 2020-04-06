"""
# SUIM-Net model for underwater image segmentation
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Usage: for academic and educational purposes only
"""
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import add, Lambda, UpSampling2D, Concatenate, ZeroPadding2D
from keras.optimizers import Adam


def RSB(input_tensor, kernel_size, filters, strides=1, skip=True):
    """ 
       A residual skip block: RSB (see Fig. 5a of the paper)
    """
    f1, f2, f3, f4 = filters
    ## sub-block1
    x = Conv2D(f1, (1, 1), strides=strides)(input_tensor)    
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    ## sub-block2
    x = Conv2D(f2, kernel_size, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    ## sub-block3
    x = Conv2D(f3, (1, 1))(x)
    x = BatchNormalization(momentum=0.8)(x)
    ## skip connetion based on the given choice 
    if skip: 
        shortcut = input_tensor
    else: 
        shortcut = Conv2D(f4, (1, 1), strides=strides)(input_tensor)
        shortcut = BatchNormalization(momentum=0.8)(shortcut)
    ## complete connection and activation
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def Suim_Encoder(inp_res=(256, 256), channels=1):
    """ 
       SUIM-Net encoder (see Fig. 5b of the paper)
    """
    im_H, im_W = inp_res
    img_input = Input(shape=(im_H, im_W, channels))
    ## encoder block 1
    x = Conv2D(64, (5, 5), strides=1)(img_input)
    enc_1 = x
    ## encoder block 2
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , strides=2)(x)
    x = RSB(x, 3, [64, 64, 128, 128], strides=2, skip=False)
    x = RSB(x, 3, [64, 64, 128, 128], skip=True)
    x = RSB(x, 3, [64, 64, 128, 128], skip=True)
    enc_2 = x 
    ## encoder block 3
    x = RSB(x, 3, [128, 128, 256, 256], strides=2, skip=False)
    x = RSB(x, 3, [128, 128, 256, 256], skip=True)
    x = RSB(x, 3, [128, 128, 256, 256], skip=True)
    x = RSB(x, 3, [128, 128, 256, 256], skip=True)
    enc_3 = x 
    ## return
    return img_input, [enc_1 , enc_2 , enc_3]


def Suim_Decoder(enc_inputs, n_classes):
    """ 
       SUIM-Net decoder (see Fig. 5b of the paper)
    """
    def concat_skip(layer_input, skip_input, filters, f_size=3):
        # for concatenation of the skip connections from encoders
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(layer_input)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u
    ## decoder block 1
    enc_1 , enc_2 , enc_3 = enc_inputs
    dec_1 = Conv2D(256, (3, 3), padding='same')(enc_3)
    dec_1 = BatchNormalization(momentum=0.8)(dec_1)
    dec_1 = UpSampling2D(size=2)(dec_1)
    # padding for matching dimenstions
    dec_1 = Lambda(lambda x : x[: , :-2 , :-2 , :  ] )(dec_1) # padding
    dec_1 = ZeroPadding2D((1,1))(dec_1)
    enc_2 = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(enc_2) # padding
    enc_2 = ZeroPadding2D((1,1))(enc_2)
    dec_1s = concat_skip(enc_2, dec_1, 256)
    ## decoder block 2
    dec_2 = Conv2D(256, (3, 3), strides=1, padding='same')(dec_1s)
    dec_2 = BatchNormalization(momentum=0.8)(dec_2)
    dec_2 = UpSampling2D(size=2)(dec_2)
    dec_2s = Conv2D(128, (3, 3), strides=1, padding='same')(dec_2)
    dec_2s = BatchNormalization(momentum=0.8)(dec_2s)
    dec_2s = UpSampling2D(size=2)(dec_2s)
    # padding for matching dimenstions
    enc_1 = ZeroPadding2D((2,2))(enc_1)
    dec_2s = concat_skip(enc_1, dec_2s, 128)
    ## decoder block 3
    dec_3 = Conv2D(128, (3, 3), padding='same')(dec_2s)
    dec_3 = BatchNormalization()(dec_3)
    dec_3s = Conv2D(64, (3, 3), padding='same')(dec_3)
    dec_3s = BatchNormalization(momentum=0.8)(dec_3s)
    ## return output layer
    out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid')(dec_3s) 
    return out


class SUIM_Net():
    """ 
       The SUIM-Net model
    """
    def __init__(self, im_res, n_classes):
        self.im_res = im_res
        self.model = self.get_model(im_res, n_classes)
        self.model.compile(optimizer=Adam(lr = 1e-4), 
                           loss = 'binary_crossentropy', 
                           metrics = ['accuracy'])

    def get_model(self, im_res, n_classes):
        img_input, features = Suim_Encoder(inp_res=im_res, channels=3)
        out = Suim_Decoder(features, n_classes) 
        return Model(input=img_input, output=out)


# sanity check
if __name__=="__main__":
    im_shape = (240, 320)
    out_channels = 5
    suim_net = SUIM_Net(im_shape, out_channels)
    print (suim_net.model.summary())




