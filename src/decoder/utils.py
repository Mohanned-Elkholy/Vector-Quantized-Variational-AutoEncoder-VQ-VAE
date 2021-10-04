import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from google.colab.patches import cv2_imshow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from src.helper_functions.utils import *

from keras.datasets import cifar10

def build_decoder(inp_shape):
    """ this function builds the decoder 
	output: the decoder model 
	"""
	# input image shape
    inp = tf.keras.layers.Input(shape=inp_shape)

    # layer 1
    x = Conv2DTranspose(64, (3, 3), activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = residual(x,64)

    # layer 2
    x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = residual(x,32)

    # layer 3
    x = Conv2DTranspose(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = residual(x,32)

    # final layer
    x = Conv2DTranspose(3, (3, 3),activation='tanh')(x)
    
    # return the model
    return tf.keras.models.Model(inp, x)

