# pytorch-VAE

Variational Autoencoder and a Disentangled version (beta-VAE). 

- [Variational Autoencdoer](./models/vae.py)

    The Variational Autoencoder is a Generative Model. Its goal is to learn the distribution of a Dataset, and then generate new (unseen) data points from the same distribution. 

- [beta Variational Autoencoder](./models/beta_vae.py)
    
    Another form of a Variational Autoencoder is the beta-VAE. The difference between the Vanilla VAE and the beta-VAE is in the loss function of the latter: The KL-Divergence term is multiplied with a hyperprameter beta. This introduces a disentanglement to the idea of the VAE, as in many cases it allows a smoother and more "continuous" transition of the output data, for small changes in the latent vector z. More information on this topic can be found in the sources section below.

## Requirements

- [PyTorch Lightning](https://www.pytorchlightning.ai/)

## Execution

#### Installation

```bash
$ pip install -r requirements.txt
```
 
#### VAE:
   ```bash
   $ python3 main.py -c <config_file_path> -v VAE
   ```
#### beta-VAE:
   ```bash
   $ python3 main.py -c <config_file_path> -v B-VAE
   ```
  

Examples of configuration files can be found in the [config](config) directory.

# Variational Autoencoder Paper

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
