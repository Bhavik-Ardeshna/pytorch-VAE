"""
Implements the Original Variational Autoencoder paper: https://arxiv.org/pdf/1312.6114.pdf
"""

import multiprocessing

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.layers import encoder, decoder, out_layer
from utils.utils import plot_multiple

class VAE(pl.LightningModule):
    
    def __init__(self, architecture, hyperparameters, dataset_info):
        super(VAE, self).__init__()
        
        self.conv_layers = architecture["conv_layers"]
        self.conv_channels = architecture["conv_channels"]
        self.conv_kernel_sizes = architecture["conv_kernel_sizes"]
        self.conv_strides = architecture["conv_strides"]
        self.conv_paddings = architecture["conv_paddings"]
        self.z_dim = architecture["z_dimension"]
        
        self.batch_size = hyperparameters["batch_size"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.scheduler_step_size = hyperparameters["epochs"] // 2

        self.dataset_method = dataset_info["ds_method"]
        self.dataset_shape = dataset_info["ds_shape"]
        self.dataset_path = dataset_info["ds_path"]
        
        # creating encoder
        self.encoder, self.encoder_shapes = encoder(architecture, self.dataset_shape)
        
        # compute the length of the output of the decoder once it has been flattened
        in_features = self.conv_channels[-1] * np.prod(self.encoder_shapes[-1][:])
        # mean and standard deviation layers
        self.mean_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)
        self.std_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)

        # use a linear layer for the input of the decoder
        self.decoder_input = nn.Linear(in_features=self.z_dim, out_features=in_features)

        # creating decoder layer
        self.decoder = decoder(architecture, self.encoder_shapes)
        
        # creating output layer
        self.out_layer = out_layer(architecture, self.dataset_shape)
        
    def _encode(self, X):
        encoded_input = self.encoder(X)
        
        # flatten so that it can be fed to the mean and standard deviation layers
        encoded_input = torch.flatten(encoded_input, start_dim=1)
        
        # compute the mean and standard deviations
        mean = self.mean_layer(encoded_input)
        std = self.std_layer(encoded_input)
        
        return mean, std
    
    def _compute_latent_vector(self, mean, std):
        
        # create normal distribution using std
        epsilon = torch.randn_like(std)
        
        return mean + epsilon * (1.0 / 2) * std
    
    def _decode(self, z):
        
        decoder_input = self.decoder_input(z)

        # convert back the shape that will be fed to the decoder
        height = self.encoder_shapes[-1][0]
        width = self.encoder_shapes[-1][1]
        decoder_input = decoder_input.view(-1, self.conv_channels[-1], height, width)

        # run through the decoder
        decoder_output = self.decoder(decoder_input)

        # run through the output layer and return
        network_output = self.out_layer(decoder_output)
        
        return network_output
    
    def forward(self, X):
        
        # encode the input to get mean and standard deviation
        mean, std = self._encode(X)
        
        # get the latent vector z by using the reparameterization trick
        z = self._compute_latent_vector(mean, std)
        
        # compute the output by propagating the latent vector through the decoder and return
        decoded_output = self._decode(z)
        
        return decoded_output, mean, std
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size,
                                                    gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        
        X, y = batch
        
        X_hat, mean, std = self(X)
        
        losses = VAE.criterion(X, X_hat, mean, std)

        return losses

    def train_dataloader(self):
       
        train_set = self.dataset_method(root=self.dataset_path, train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())

        self.train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=multiprocessing.cpu_count() // 2)
        
        return self.train_loader


    def test_dataloader(self):
        
        test_set = self.dataset_method(root=self.dataset_path, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

        self.test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True,
                                      num_workers=multiprocessing.cpu_count() // 2)
        return self.test_loader
    
    def test_step(self, batch, batch_idx):
        X, y = batch

        X_hat, mean, std = self(X)
        
        losses = VAE.criterion(X, X_hat, mean, std)
        

        mse_loss_func = torch.nn.MSELoss()
        mse_loss = mse_loss_func(X, X_hat)

        for key, value in losses.items():
            self.log(f'{key}', value.item())
        self.log('mse_loss', mse_loss.item())

        return losses, mse_loss
    
    def sample(self, n):
        z = torch.randn(n*n, self.z_dim)

        samples = self._decode(z)

        cmap = None
        
        if (self.dataset_shape[0] == 3):
            cmap = 'viridis'
        
        elif (self.dataset_shape[0] == 1):
            cmap = 'gray'
        
    
    def reconstruct(self, n):
        
        tensors = []
        img_count = 0
        while n * n > img_count:
            batch, y = next(iter(self.test_loader))
            img_count += len(batch)
            tensors.append(batch)

        X = torch.cat(tensors, dim=0)

        X_hat, mean, std = self(X)
        min_imgs = min(n, len(X))

        cmap = None
        if (self.dataset_shape[0] == 3):
            cmap = 'viridis'
        elif (self.dataset_shape[0] == 1):
            cmap = 'gray'

        # plot the images and their reconstructions
        plot_multiple(X_hat.detach().numpy(), min_imgs, self.dataset_shape, cmap)

    @staticmethod
    def _data_fidelity_loss(X, X_hat, eps=1e-10):
        """
        E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))] = sum(x * log(x_hat) + (1 - x) * log(1 - x_hat))

            which is basically a Cross Entropy Loss.

        This method computes the Data Fidelity term of the loss function. A small positive double
        epsilon is added inside the logarithm to make sure that we don't get log(0).
        """
        data_fidelity = torch.sum(X * torch.log(eps + X_hat) + (1 - X) * torch.log(eps + 1 - X_hat),
                                  axis=[1, 2, 3])
        return data_fidelity 
    
    @staticmethod
    def _kl_divergence_loss(mean, std):
        """
        D_{KL}[q_{phi}(z | x) || p_{theta}(x)] = (1/2) * sum(std + mean^2 - 1 - log(std))

            In the above equation we substitute std with e^{std} to improve numerical stability.

        This method computes the KL-Divergence term of the loss function. It substitutes the
        value of the standard deviation layer with exp(standard deviation) in order to ensure
        numerical stability.
        """
        kl_divergence = (1 / 2) * torch.sum(torch.exp(std) + torch.square(mean) - 1 - std, axis=1)
        return kl_divergence
    
    @staticmethod
    def criterion(X, X_hat, mean, std):
        """
        This method computes the loss of the VAE using the formula:

            L(x, x_hat) = - E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))]
                          + D_{KL}[q_{phi}(z | x) || p_{theta}(x)]

        Intuitively, the expectation term is the Data Fidelity term, and the second term is a
        regularizer that makes sure the distribution of the encoder and the decoder stay close.
        """
        # get the 2 losses
        data_fidelity_loss = VAE._data_fidelity_loss(X, X_hat)
        kl_divergence_loss = VAE._kl_divergence_loss(mean, std)

        # add them to compute the loss for each training example in the mini batch
        loss = -data_fidelity_loss + kl_divergence_loss

        # place them all inside a dictionary and return it
        losses = {"data_fidelity": torch.mean(data_fidelity_loss),
                  "kl-divergence": torch.mean(kl_divergence_loss),
                  "loss": torch.mean(loss)}
        return losses
    
