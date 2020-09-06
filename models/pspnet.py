import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import keras.backend as K

from mobilenet import get_mobilenet_encoder
from resnet50 import get_resnet50_encoder
"""
Code taken from: 
https://github.com/fchollet/deep-learning-models
"""

IMAGE_ORDERING = 'channels_last'

if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1


def resize_image( inp ,  s , data_format ):
	try:
		
		return Lambda( lambda x: K.resize_images(x, 
			height_factor=s[0], 
			width_factor=s[1], 
			data_format=data_format , 
			interpolation='bilinear') )( inp )

	except Exception as e:
		# if keras is old , then rely on the tf function ... sorry theono/cntk users . 
		assert data_format == 'channels_last'
		assert IMAGE_ORDERING == 'channels_last'
		return Lambda( 
			lambda x: tf.image.resize_images(
				x , ( K.int_shape(x)[1]*s[0] ,K.int_shape(x)[2]*s[1] ))  
			)( inp )


def pool_block( feats , pool_factor ):


	if IMAGE_ORDERING == 'channels_first':
		h = K.int_shape( feats )[2]
		w = K.int_shape( feats )[3]
	elif IMAGE_ORDERING == 'channels_last':
		h = K.int_shape( feats )[1]
		w = K.int_shape( feats )[2]

	pool_size = strides = [int(np.round( float(h) /  pool_factor)), int(np.round(  float(w )/  pool_factor))]

	x = AveragePooling2D(pool_size , data_format=IMAGE_ORDERING , strides=strides, padding='same')( feats )
	x = Conv2D(512, (1 ,1 ), data_format=IMAGE_ORDERING , padding='same' , use_bias=False )( x )
	x = BatchNormalization()(x)
	x = Activation('relu' )(x)

	x = resize_image( x , strides , data_format=IMAGE_ORDERING ) 

	return x




def _pspnet( n_classes , encoder ,  input_height=384, input_width=576  ):
	#assert input_height%192 == 0
	#assert input_width%192 == 0
	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels 
	o = f5

	pool_factors = [ 1, 2 , 3 , 6 ]
	pool_outs = [o ]

	for p in pool_factors:
		pooled = pool_block(  o , p  )
		pool_outs.append( pooled )
	
	o = Concatenate( axis=MERGE_AXIS)(pool_outs )
	for i in range(1, 5):
		o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
		o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
		o = (Conv2D(3072/(2*i), (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
		o = (BatchNormalization())(o)
	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	out = Conv2D(n_classes, 3, padding='same', activation = 'sigmoid')(o)
	model = Model(img_input, out)
	return model


def mobilenet_pspnet(n_classes, input_height=384, input_width=384 ):
	model =  _pspnet(n_classes, 
                         get_mobilenet_encoder,  
                         input_height, 
                         input_width)
	model.compile(optimizer = Adam(lr = 1e-4), 
                      loss = 'binary_crossentropy', 
                      metrics = ['accuracy'])
	return model




