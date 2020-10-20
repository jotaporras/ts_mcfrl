import argparse
import logging
import random
from collections import namedtuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from envs import network_flow_env_builder
from pytorch_lightning import metrics
from pytorch_lightning.loggers import WandbLogger
from shipping_allocation import EnvironmentParameters
from torch import nn

# Named tuple for storing experience steps gathered in training
from experiments_v2.ptl_agents import (
    DQNLightning,
    MyPrintingCallback,
    ShippingFacilityEnvironmentStorageCallback,
    WandbDataUploader,
)

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class CustomerDQN(nn.Module):
    """
    Simple MLP network.
    From

    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, num_dcs, hidden_size: int = 128):
        super(CustomerDQN, self).__init__()
        self.obs_size = obs_size
        self.num_dcs = num_dcs
        self.net = nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

        # Definition of the 4 extra metadata neurons (check network_flow_env to verify)
        # ship_id = latest_open_order.shipping_point.node_id
        # customer_id = latest_open_order.customer.node_id
        # current_t = state['current_t']
        # delivery_t = latest_open_order.due_timestep
        # metadata = np.array([[ship_id, customer_id, current_t, delivery_t]]).transpose()
        self.customer_metadata_neuron = -3


    def forward(self, x):
        # Convert vector into one hot encoding of the customer.
        with torch.no_grad():
            xp = metrics.functional.to_onehot(x[:,self.customer_metadata_neuron]-self.num_dcs, num_classes=self.obs_size)

        return self.net(xp.float())


class DQNLightningOneHot(DQNLightning):
    """
    DQN But with a network that does one hot encoding.
    """

    def __init__(
        self, hparams: argparse.Namespace, environment_parameters: EnvironmentParameters
    ) -> None:
        super(DQNLightningOneHot,self).__init__(hparams, environment_parameters)

        # Observation space for this network is the number of customers (onehot).
        obs_size = self.env.environment_parameters.network.num_customers
        n_actions = self.env.action_space.n

        num_dcs = self.env.environment_parameters.network.num_dcs
        self.net = CustomerDQN(obs_size, n_actions, num_dcs)
        self.target_net = CustomerDQN(obs_size, n_actions, num_dcs)


def main() -> None:
    config_dict = {
        "env": {
            "num_dcs": 3,
            "num_customers": 100,
            "num_commodities": 35,
            "orders_per_day": int(100 * 0.05),
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "replay_size": 150,
            "warm_start_steps": 150,  # apparently has to be smaller than batch size
            "max_episodes": 35,  # to do is this num episodes, is it being used?
            "episode_length": 30,  # todo isn't this an env thing?
            "batch_size": 30,
            "gamma": 0.99,
            "eps_end": 1.0,  # todo consider keeping constant to start.
            "eps_start": 0.99,  # todo consider keeping constant to start.
            "eps_last_frame": 1000,  # todo maybe drop
            "sync_rate": 2,  # Rate to sync the target and learning network.
            "lr": 1e-2,
        },
        "seed": 0,
    }

    torch.manual_seed(config_dict["seed"])
    np.random.seed(config_dict["seed"])
    random.seed(config_dict["seed"])  # not sure if actually used
    np.random.seed(config_dict["seed"])

    run = wandb.init(config=config_dict)

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    experiment_name = "dqn_onehot_few_warehouses_v4"
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            # "debug"
            "experiment"
        ],
        log_model=True,
    )

    wandb_logger.log_hyperparams(dict(config))

    environment_parameters = network_flow_env_builder.build_network_flow_env_parameters(
        environment_config, hparams["episode_length"], order_gen="biased"
    )

    model = DQNLightningOneHot(hparams, environment_parameters)

    trainer = pl.Trainer(
        max_epochs=hparams["max_episodes"] * hparams["replay_size"],
        early_stop_callback=False,
        val_check_interval=100,
        logger=wandb_logger,
        log_save_interval=1,
        row_log_interval=1,  # the default of this may leave info behind.
        callbacks=[
            MyPrintingCallback(),
            ShippingFacilityEnvironmentStorageCallback(
                experiment_name,
                base="data/results/",
                experiment_uploader=WandbDataUploader(),
            ),
        ],
    )

    trainer.fit(model)


if __name__ == "__main__":
    logging.root.level = logging.INFO
    main()