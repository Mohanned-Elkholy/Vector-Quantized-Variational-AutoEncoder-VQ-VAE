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
from keras.datasets import cifar10

def get_the_data_normalized():
    """ 
    This function imports the data from keras. Later it normalizes the data between -1 and 1.
    output:
    training data: numpy array of the data of the shape (-1,32,32,3)
    testing data: numpy array of the data of the shape (-1,32,32,3)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize the data (between -1 and 1)
    x_train = np.reshape(x_train,(-1,32,32,3))/127.5-1
    x_test = np.reshape(x_test,(-1,32,32,3))/127.5-1
    return x_train,x_test