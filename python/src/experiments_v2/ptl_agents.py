import argparse
import logging
import random
from typing import List
from typing import Tuple

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from envs import ShippingFacilityEnvironment, network_flow_env_builder
from pytorch_lightning.loggers import WandbLogger
from shipping_allocation import EnvironmentParameters
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Named tuple for storing experience steps gathered in training
from dqn.dqn_common import Experience, RLDataset, ReplayBuffer
from experiments_v2.base_ptl_agent_runner import DQNLightning
from experiments_v2.ptl_callbacks import WandbDataUploader, MyPrintingCallback, \
    ShippingFacilityEnvironmentStorageCallback

# Default big-ish
# config_dict = {
#     "env": {
#         "num_dcs": 3,
#         "num_customers": 100,
#         "num_commodities": 35,
#         "orders_per_day": int(100 * 0.05),
#         "dcs_per_customer": 2,
#         "demand_mean": 500,
#         "demand_var": 150,
#         "num_steps": 30,  # steps per episode
#         "big_m_factor": 10000 # how many times the customer cost is the big m.
#     },
#     "hps": {
#         "env": "shipping-v0", #openai env ID.
#         "replay_size": 150,
#         "warm_start_steps": 150, # apparently has to be smaller than batch size
#         "max_episodes": 500, # to do is this num episodes, is it being used?
#         "episode_length": 30, # todo isn't this an env thing?
#         "batch_size": 30,
#         # "gamma": 0.99,
#         # "lr": 1e-2,
#         "eps_end": 1.0, #todo consider keeping constant to start.
#         "eps_start": 0.99, #todo consider keeping constant to start.
#         "eps_last_frame": 1000, # todo maybe drop
#         "sync_rate": 2, # Rate to sync the target and learning network.
#     },
#     "seed": 0
# }

# Debug dict
config_dict = {
    "env": {
        "num_dcs": 3,
        "num_customers": 100,
        "num_commodities": 3,
        "orders_per_day": 1,
        "dcs_per_customer": 2,
        "demand_mean": 500,
        "demand_var": 150,
        "num_steps": 10,  # steps per episode
        "big_m_factor": 10000
    },
    "hps": {
        "env": "shipping-v0", #openai env ID.
        "replay_size": 30,
        "warm_start_steps": 30, # apparently has to be smaller than batch size
        "max_episodes": 20, # to do is this num episodes, is it being used?
        "episode_length": 30, # todo isn't this an env thing?
        #"batch_size": 30,
        "batch_size": 2,
        # tuneable need to be at root #TODO CHANGE ALL CONFIGS IF WORKS
        "lr": 1e-2,
        "gamma": 0.99,
        "eps_end": 1.0, #todo consider keeping constant to start.
        "eps_start": 0.99, #todo consider keeping constant to start.
        "eps_last_frame": 1000, # todo maybe drop
        "sync_rate": 2, # Rate to sync the target and learning network.

    },
    "seed":0,
}
#experiment_name = "dqn_few_warehouses_bigmreward"
experiment_name = "debug_nodebalance"



def main() -> None:
    run = wandb.init(config=config_dict)

    torch.manual_seed(config_dict['seed'])
    np.random.seed(config_dict['seed'])
    random.seed(config_dict['seed']) # not sure if actually used
    np.random.seed(config_dict['seed'])

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    # TODO Hotfix because wandb doesn't support sweeps.
    if "lr" in config:
        hparams["lr"] = config.lr
        hparams["gamma"] = config.gamma

    print("CONFIG CHECK FOR SWEEP")
    logging.warning(hparams['lr']) #todo aqui quede make sweep work something with imports.
    logging.warning(hparams['gamma'])


    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            "debug"
            # "experiment"
        ],
        log_model=True

    )

    wandb_logger.log_hyperparams(dict(config))

    environment_parameters = network_flow_env_builder.build_network_flow_env_parameters(
        environment_config,
        hparams['episode_length'],
        order_gen='biased'
    )

    model = DQNLightning(hparams, environment_parameters)


    trainer = pl.Trainer(
        max_epochs=hparams['max_episodes']*hparams['replay_size'],
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

    trainer.fit(model)


if __name__ == '__main__':
    logging.root.level = logging.DEBUG
    main()