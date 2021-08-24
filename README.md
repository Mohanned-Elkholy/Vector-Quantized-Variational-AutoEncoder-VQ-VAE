# Vector-Quantized-Variational-AutoEncoder-(VQ-VAE)
This is keras implementation of the Vector Quantized Variational Auto Encoder on Cifar10 data set. You can know more about Vector quantized Variational Auto Encoders here (https://arxiv.org/pdf/1711.00937.pdf). 
![image](https://user-images.githubusercontent.com/47930821/130600487-83cdb3e5-67a7-4f06-a97a-9bb7fd7adcd3.png)

---

# Vanilla Variational AutoEncoder
The challenge with variational AutoEncoder is the blurry reconstruction of the image. This issue arises from the sampling layer that was trained by choosing a random unit variance.

---

# VQ-VAE
In this model, the sampling layer is rather discretized. To further illustrate this, the decoder samples from an embedding vector space. In other words, each channel vector at the encoder's output is replaced by the closest vector in the embedding space. Therefore, the sampling is quantized.
![image](https://user-images.githubusercontent.com/47930821/130602616-cf52fa8e-5c33-4e8a-bc22-320f5885b66e.png)

# The Non-differentiability of the VQ-VAE
The sampling layer chooses the closest embedding vector. In other words, it uses the argmin function. Unfortunately, this is not differentiable, and so, the gradients can't move to the encoder. This challenge is overcome by copying the gradients at the beginning of the decoder to the end of the encoder (skip the embedding space). 
![image](https://user-images.githubusercontent.com/47930821/130603643-1cdffa28-4d2b-4b76-9d69-73de7c2abe17.png)

On the other hand, the embedding space would be better if it is close to the encoder output to avoid unexpected divergence in the reconstruction. Therefore, another term should be added to the loss function.

![image](https://user-images.githubusercontent.com/47930821/130603862-345648b0-df02-4564-b0bf-166646ade86b.png)

# Putting all together
Now, the embedding vectors can be differentiated and the gradients can also pass through them. It only needs a reconstruction loss to make the VQ-VAE and end-to-end trainable model. This is the total loss function. 
![image](https://user-images.githubusercontent.com/47930821/130604166-0c6435c9-d6b1-48d9-877f-7a6a94c033e4.png)
Usually Beta is set to 0.25 according to the paper.

---
# Prerequisites
1- python3 

2- CPU or NVIDIA GPU + CUDA CuDNN (GPU is recommended for faster inversion)

---

# Install dependencies
In this repo, a pretrained biggan in a specified library
```python
pip install torch torchvision matplotlib lpips numpy nltk cv2 pytorch-pretrained-biggan
```
# Training
#provide image to work on
```python
python train.py  --num_epochs 2000 --learning_rate 0.007 
```
---
