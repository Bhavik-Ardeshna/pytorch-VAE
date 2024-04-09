"""
Implements the betaVAE from the paper: https://arxiv.org/pdf/1606.05579.pdf
"""

import torch
import torch.nn as nn

from models.vae import VAE

class betaVAE(VAE):
    
    def __init__(self, architecture, hyperparameters, dataset_info):
        super(betaVAE, self).__init__(architecture, hyperparameters, dataset_info)
        
        # store the value of beta in the class as it exists only in this VAE variation
        self.beta = hyperparameters["beta"]
        
    @staticmethod
    def criterion(X, X_hat, mean, std):
        """
        This method computes the loss of the B-VAE using the formula:

            L(x, x_hat) = - E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))]
                          + beta * D_{KL}[q_{phi}(z | x) || p_{theta}(x)]

        Intuitively, the expectation term is the Data Fidelity term, and the second term is a
        regularizer that makes sure the distribution of the encoder and the decoder stay close.
        """
        data_fidelity_loss = VAE._data_fidelity_loss(X, X_hat)
        kl_divergence_loss = VAE._kl_divergence_loss(mean, std)

        loss = -data_fidelity_loss + self.beta * kl_divergence_loss

        losses = {"data_fidelity": torch.mean(data_fidelity_loss),
                  "kl-divergence": torch.mean(kl_divergence_loss),
                  "beta_kl-divergence": self.beta * torch.mean(kl_divergence_loss),
                  "loss": torch.mean(loss)}
        return losses
