import torch
import torch.nn as nn
from pytorch_lightning import Trainer

import logging

from utils.interface import parse_config_file, parse_cmd_args
from utils.utils import filepath_is_not_valid, prepare_dataset
from models.vae import VAE
from models.beta_vae import betaVAE


def main(args):
    """ main() driver function """

    # Parameters parsing
    if filepath_is_not_valid(args.config):
        logging.error("The path {} is not a file. Aborting..".format(args.config))
        exit()

    configuration, architecture, hyperparameters = parse_config_file(args.config, args.variation)
    dataset_info = prepare_dataset(configuration)
    if (dataset_info is None):
        exit()

    # Initialization
    model = None
    if (args.variation == "VAE"):
        model = VAE(architecture, hyperparameters, dataset_info)
    elif (args.variation == "B-VAE"):
        model = betaVAE(architecture, hyperparameters, dataset_info)

    # here you can change the gpus parameter into the amount of gpus you want the model to use
    trainer = Trainer(max_epochs = hyperparameters["epochs"], accelerator="auto", fast_dev_run=False)

    # Training and testing
    trainer.fit(model)
    result = trainer.test(model)
    # Model needs to be transferred to the cpu as sample and reconstruct are custom methods
    model = model.cpu()
    model.sample(5)
    model.reconstruct(5)

if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_cmd_args()
    # call the main() driver function
    main(args)
    print("\n")
