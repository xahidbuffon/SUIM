import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from models.resnet50 import get_resnet50_encoder, vanilla_encoder

IMAGE_ORDERING = 'channels_last'

def segnet_decoder(f, n_classes, n_up=2):
	assert n_up >= 2
	o = f
	o = (ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	for _ in range(n_up):
		o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
		o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
		o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
		o = (BatchNormalization())(o)
	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	out = Conv2D(n_classes, 3, padding='same', activation = 'sigmoid')(o)
	return out


def _segnet(n_classes, encoder, input_height, input_width, encoder_level=3):	
	img_input, levels = encoder(input_height=input_height,  
                                    input_width=input_width, 
                                    channels=3)
	feat = levels[encoder_level]
	out = segnet_decoder(feat, n_classes, n_up=2)
	model = Model(input=img_input, output=out)
	model.compile(optimizer = Adam(lr = 1e-4), 
                      loss = 'binary_crossentropy', 
                      metrics = ['accuracy'])
	return model


def resnet50_segnet(n_classes=2, input_height=224, input_width=224, encoder_level=3):
	model = _segnet(n_classes, 
                        get_resnet50_encoder, 
                        input_height=input_height, 
                        input_width=input_width, 
                        encoder_level=encoder_level)
	return model


def cnn_segnet(n_classes=2, input_height=224, input_width=224, encoder_level=3):
	model = _segnet(n_classes, 
                        vanilla_encoder, 
                        input_height=input_height, 
                        input_width=input_width, 
                        encoder_level=encoder_level)
	return model


