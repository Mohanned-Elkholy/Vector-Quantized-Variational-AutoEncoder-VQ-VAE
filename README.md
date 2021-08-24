# Vector-Quantized-Variational-AutoEncoder-(VQ-VAE)
This is keras implementation of the Vector Quantized Variational Auto Encoder on Cifar10 data set. You can know more about Vector quantized Variational Auto Encoders here (https://arxiv.org/pdf/1711.00937.pdf). 
![image](https://user-images.githubusercontent.com/47930821/130600487-83cdb3e5-67a7-4f06-a97a-9bb7fd7adcd3.png)

---

# Vanilla Variational AutoEncoder
The challenge with variational AutoEncoder is the blurry reconstruction of the image. This issue arises from the sampling layer that was trained by choosing a random unit variance.

---

# VQ-VAE
In this model, the sampling layer is rather discretized. To further illustrate this, the decoder samples from an embedding vector space. In other words, each channel vector at the encoder's output is replaced by the closest vector in the embedding space. Therefore, the sampling is quantized.
