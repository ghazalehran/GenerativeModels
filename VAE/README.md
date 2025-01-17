# Variational Autoencoder (VAE)

This repository implements a Variational Autoencoder (VAE) using PyTorch. The repository contains both the VAE model implementation and a tutorial for training and testing the VAE.

## Overview

### VAE Model
The VAE consists of two main components:
- **Encoder**: Encodes the input image into a latent space representation.
- **Decoder**: Reconstructs the image from the latent space representation.
- **Reparameterization Trick**: This trick enables the model to backpropagate through the sampling process in the latent space.

The loss function combines:
- **Reconstruction Loss**: Measures how well the model reconstructs the input.
- **KL Divergence**: Regularizes the latent space, encouraging it to follow a standard normal distribution.

### Getting Started With VAE
The repository also includes a step-by-step **tutorial** that demonstrates how to implement and train the VAE on the **SVHN dataset**. It covers:
- **Data Preparation**: Loading and preprocessing the SVHN dataset.
- **Model Definition**: Implementing the encoder, decoder, and VAE.
- **Training**: Training the VAE using both the reconstruction loss and KL divergence.
- **Visualization**: Visualizing the results, including reconstructed images, generated samples, and latent space interpolation.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

You can install the necessary dependencies using `pip`:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/vae-svhn.git
cd vae-svhn
```

2. Training VAE on SVHN Dataset:

- Open and run the tutorial notebook Getting_Started_With_VAE.ipynb. This notebook includes the full code for:
    - Data preparation
    - Model training
    - Visualization of generated images
- The model is trained for 30 epochs on the SVHN dataset


3. The VAE will be trained on the SVHN dataset, and the generated images, log-likelihood estimates, and latent space interpolations will be saved to the results/ folder.

## VAE Model Architecture

### Encoder
The encoder is a convolutional neural network (CNN) that extracts a latent space representation from the input image.

### Decoder
The decoder takes the latent space representation and reconstructs the image using deconvolution (transposed convolution) layers.

### VAE Loss
The VAE loss is a combination of:

- Reconstruction loss (mean squared error between the original and reconstructed images)
- KL divergence (regularizing the latent space to follow a standard normal distribution)

### Training the VAE
In the tutorial, the following hyperparameters are used for training:

- Latent space dimension (z_dim): 32
- Batch size: 64
- Learning rate: 1e-4
- Epochs: 30
- During training, the VAE model learns to encode the input images into a lower-dimensional latent space, and decode them back into realistic images.

### Results and Visualization
The tutorial also demonstrates various visualizations:

- Generated Samples: Visualizing samples generated from the latent space.
- Reconstruction: Comparing the original and reconstructed images.
- Latent Space Interpolation: Interpolating between two latent vectors to visualize smooth transitions.
- Log Likelihood: Estimating the log-likelihood of the test dataset.
  
### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
The SVHN dataset is provided by SVHN.

VAE_Model.py is the completed assignment of Representation Learning course  by Dr. Aaron Courville, IFT6135-H2023

This implementation follows the general structure of a Variational Autoencoder for image generation.
