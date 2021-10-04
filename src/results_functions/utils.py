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
import cv2
import matplotlib.gridspec as gridspec
from keras.datasets import cifar10


def get_indices_from_image_batch(image_batch,encoder,vq_model):
    encoder_output = encoder.predict(image_batch)
    vq_output = vq_model.predict(encoder_output)
    return vq_output[-1]

def reconstruct_image_from_indices(indices_batch,vq_model,decoder):
    output_vq = vq_model.layers[2].quantize(indices_batch)[0]
    return decoder.predict(output_vq)

def plot_for_testing(x_train,encoder,vq_model,decoder):
    for i in range(50):
        indices = get_indices_from_image_batch(x_train[i:i+1],encoder,vq_model)
        out = reconstruct_image_from_indices(indices,vq_model,decoder)
        plt.imshow(x_train[i]*0.5+0.5)
        plt.show()
        plt.imshow(out[0]*0.5+0.5)
        plt.show()


def showReconstructedRealBatch(x_train,encoder,vq_model,decoder,epoch):
    """
    Show bathches of real and reconstructed images
    """
    def show_batch_image(img):
        img = np.reshape(img,(len(img),32,32,3))
        if np.min(img)<-0.9:
            img = (img+1)*0.495
        num = len(img)
        ax = np.ceil(np.sqrt(num)) 
        ay = np.rint(np.sqrt(num)) 
        fig =plt.figure()
        for i in range(1,num+1):
            sub = fig.add_subplot(ax,ay,i)
            sub.axis('off')
            sub.imshow(img[i-1])
        plt.show()
        plt.close()

    def show_batch_image(images,status,epoch=0):
        """
        Show bathches of real and reconstructed images
        """
        # convert to numpy array
        img = np.array(images).reshape(-1,32,32,3)
        # make sure it is normalized
        if np.min(img)<-0.9:
            img = (img+1)*0.495
        # create resized batch
        img = np.array([cv2.resize(i,(32,32)) for i in img])
        num = len(img)
        # set the coordinates for the figure
        ax = np.ceil(np.sqrt(num)) 
        ay = np.rint(np.sqrt(num)) 
        fig = plt.figure(figsize=(64, 64))
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(img):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample * 0.5 + 0.5)
        plt.savefig(f'outputs/{epoch}_{status}.png', bbox_inches='tight')
        plt.close(fig)

        
    fakeList=[]
    for i in range(36):
        indices = get_indices_from_image_batch(x_train[i:i+1],encoder,vq_model)
        out = reconstruct_image_from_indices(indices,vq_model,decoder)[:,:,:,[2,1,0]]
        fakeList.append(out)
    
    print('Reconstructed')
    show_batch_image(fakeList,'reconstructed',epoch)
    print('Real')
    show_batch_image(x_train[:36][:,:,:,[2,1,0]],'real',epoch)
