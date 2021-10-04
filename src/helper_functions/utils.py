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
from src.results_functions.utils import *

def residual(x,original_channels):
    """ This function takes the input x and the channels to be in the conv layers
	input: 
	x: the tensor in the forward keras model
	original_channels: number of channels in the conv layers
	output:
	the tensor x after passing through this block
	"""
    # left block (the conv block)
    x2 = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = Conv2D(original_channels, (3, 3), activation='relu',padding='same')(x)
    x2 = tf.keras.layers.BatchNormalization()(x2)

    # right block (the skip block)
    x=x

    # summing the conv and the skip
    output = tf.keras.layers.Concatenate()([x,x2])
    return output


# VQ layer.
class VQVAELayer(Layer):

    def build(self, input_shape=(24, 24, 1),embedding_dim=64,num_embeddings=100):
        # Add embedding weights.
        self.w = self.add_weight(
                                  shape=(embedding_dim, num_embeddings),
                                  initializer='uniform',
                                  trainable=True)

        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def get_indices(self,x,embedding_dim=64,num_embeddings=100):
        """ This function gets the embedding vector indices of the input x """        
        # flatten the inputs
        flat_inputs = tf.keras.backend.reshape(x, (-1, embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (tf.keras.backend.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * tf.keras.backend.dot(flat_inputs, self.w)
                     + tf.keras.backend.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = tf.keras.backend.argmax(-distances, axis=1)
        encodings = tf.keras.backend.one_hot(encoding_indices, num_embeddings)
        encoding_indices = tf.keras.backend.reshape(encoding_indices, tf.keras.backend.shape(x)[:-1])
        return encoding_indices

    def call(self, x):
        # Flatten input except for last dimension.
        encoding_indices = self.get_indices(x)
        quantized,encoding_indices = self.quantize(encoding_indices)

        # Metrics.
        return quantized,encoding_indices

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = tf.keras.backend.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices,name='hello'),encoding_indices



def vq_loss(commitment_cost, quantized, x_inputs):
    """ this function returns the loss of the embedding vector """
	# this line approximates the inputs to the embedding vector and copies the gradients to the encoder implicitly
    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x_inputs) ** 2)

    # this line approximates the embedding vector to the inputs
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(x_inputs)) ** 2)

    # combine both and return the loss
    loss = q_latent_loss + commitment_cost * e_latent_loss
    return loss


def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    """ this function wraps the loss function and return the function itself """
    def vq_vae_loss(x, x_hat):

    	# reconstruction loss
        recon_loss = losses.mse(x, x_hat) / data_variance

        # commitment loss
        VQ_loss = vq_loss(commitment_cost, quantized, x_inputs)
        return recon_loss + VQ_loss 

    return vq_vae_loss


def build_vector_quantizer(inp_shape=(24, 24, 1),embedding_dim=64):
    """ This function builds the VQ-VAE layer as a keras model """

    v = tf.keras.layers.Input(shape=inp_shape)
    enc = Conv2D(embedding_dim, kernel_size=(1, 1), strides=(1, 1), name="pre_vqvae")(v)
    enc_inputs = enc
    the_vq_layer = VQVAELayer()
    enc,encoding_indices = the_vq_layer(enc)
    x = enc_inputs + tf.keras.backend.stop_gradient(enc - enc_inputs)
    return tf.keras.models.Model(v, [x,enc,enc_inputs,encoding_indices])

	# the input
    v = tf.keras.layers.Input(shape=inp_shape)

    # allign the number of channels with the embedding vector
    emb_inputs = Conv2D(embedding_dim, kernel_size=(1, 1), strides=(1, 1))(v)

    # construct the layer
    the_vq_layer = VQVAELayer(inp_shape,64)
    print(emb_inputs)

    # get the embedding outputs
    emb_outputs,encoding_indices = the_vq_layer(emb_inputs)

    # stop the gradient with the difference and add the embedding inputs. This copies the gradients on the embedding inputs
    # this line is responsible for copying the gradients from the decoder to the encoder
    x = emb_inputs + tf.keras.backend.stop_gradient(emb_outputs - emb_inputs)

    # return the model
    return tf.keras.models.Model(v, [x,emb_outputs,emb_inputs,encoding_indices])

def build_vq_vae(encoder,decoder,vq_model,commitment_cost=0.25,train=True):
    """ This function builds thw whole VQ-VAE model """

    input_img = tf.keras.layers.Input(shape=(32, 32, 3))
    
    # construct the encoder
    enc = encoder(input_img)

    # get the output of the vq_model
    vq,enc,enc_inputs,indices_enc = vq_model(enc)

    # get the output of the decoder
    x = decoder(vq)

    # construct the model
    vqvae = tf.keras.models.Model(input_img, x)

    # append the loss of the VQ to the model
    the_loss_of_vq = vq_loss(commitment_cost, enc, enc_inputs)
    vqvae.add_loss(the_loss_of_vq)

    # set the optimizer and compile the model
    optimizer = keras.optimizers.adam_v2.Adam()
    vqvae.compile(loss=['mae','mse'], optimizer=optimizer,experimental_run_tf_function=False)
    return vqvae


def train(x_train,vqvae,epochs,encoder,vq_model,decoder):
    """ train the VQ-VAE """
    for anEpoch in (range(100)):
        print('Epoch number '+str(anEpoch))
        vqvae.fit(x_train,x_train,epochs=1,validation_split=0.05,batch_size=128,shuffle=True)
        showReconstructedRealBatch(x_train,encoder,vq_model,decoder,anEpoch)

