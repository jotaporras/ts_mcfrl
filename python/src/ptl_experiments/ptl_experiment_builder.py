
import os
import random
import shutil
from collections import deque

import torch
import wandb
from envs import ShippingFacilityEnvironment, network_flow_env_builder
from envs.network_flow_env import ActualOrderGenerator, DirichletInventoryGenerator
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from shipping_allocation import EnvironmentParameters
from torch import nn
import numpy as np
import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import gym
import torch
from typing import Tuple

from collections import namedtuple
import logging

import numpy as np
import argparse
from torch.utils.data.dataset import IterableDataset

from ptl_agents import DQNLightning


def build_experiment_trainer(config_dict,mode="experiment"):
    """
    :param config_dict: configuration dictionary for experiment
    :param mode: "experiment or debug"
    :return: the PTL Module
    """
    # Initialize seeds for reproducibility
    torch.manual_seed(config_dict['seed'])
    np.random.seed(config_dict['seed'])
    random.seed(config_dict['seed']) # not sure if actually used
    np.random.seed(config_dict['seed'])

    # Initialize wandb
    run = wandb.init(config=config_dict)

    # extract config subdictionaries.
    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    # Initialize PTL W&B Logger
    experiment_name = "dqn_few_warehouses_v4__demandgen_biased"
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            # "debug"
            "experiment"
        ],
        log_model=True

    )
    wandb_logger.log_hyperparams(dict(config))


    # Initialize
    environment_parameters = network_flow_env_builder.build_network_flow_env_parameters(
        environment_config,
        hparams['episode_length'],
        order_gen='biased'
    )

    model = DQNLightning(hparams, environment_parameters)


    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        early_stop_callback=False,
        val_check_interval=100,
        logger=wandb_logger,
        log_save_interval=1,
        row_log_interval=1, # the default of this may leave info behind.
        callbacks=[
            MyPrintingCallback(),
            ShippingFacilityEnvironmentStorageCallback(experiment_name,base="data/results/",experiment_uploader=WandbDataUploader())
        ]
    )