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
from google.colab.patches import  cv2_imshow
from src.data_manipulation.utils import *
from src.encoder.utils import *
from src.decoder.utils import *
from src.helper_functions.utils import *
from src.results_functions.utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_embeddings',type=int, required = True)
parser.add_argument('--embedding_dim',type=int,required = True)
parser.add_argument('--commitment_cost',type=float,required = True)

opt = parser.parse_args()

# VQ-VAE Hyper Parameters.
num_embeddings=opt.num_embeddings # Length of embedding vectors.
embedding_dim = opt.embedding_dim # Number of embedding vectors (high value = high bottleneck capacity).
commitment_cost = opt.commitment_cost # Controls the weighting of the loss terms.


x_train,x_test = get_the_data_normalized()
epochs = 1000 
batch_size = 64
validation_split = 0.1


encoder = build_encoder()
decoder = build_decoder((24, 24, embedding_dim))
vq_model = build_vector_quantizer()
vqvae = build_vq_vae(encoder,decoder,vq_model,commitment_cost)
print(vqvae.summary())

train(x_train,vqvae,1000,encoder,vq_model,decoder)