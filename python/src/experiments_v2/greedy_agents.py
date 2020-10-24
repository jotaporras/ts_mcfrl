import logging
import random
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from envs import network_flow_env_builder
from pytorch_lightning.loggers import WandbLogger
from shipping_allocation.envs.network_flow_env import (
    EnvironmentParameters,
    ShippingFacilityEnvironment
)
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, IterableDataset

import agents
from agents import RandomAgent, Agent
from experiments_v2.ptl_agents import ShippingFacilityEnvironmentStorageCallback
from experiments_v2.ptl_callbacks import MyPrintingCallback, WandbDataUploader


class ShippingFacilityEpisodesDataset(IterableDataset):
    """
        Simple dataset to guide the PTL training based on the number of steps we're going to run.
        It's just a fancy iterator.
    """

    def __init__(self, num_steps, orders_per_day) -> None:
        self.num_steps = num_steps
        self.orders_per_day = orders_per_day

    def __iter__(self) -> Tuple:
        for step in range(self.num_steps):
            for num_order in range(self.orders_per_day):
                ep_start = step==0 and num_order==0
                yield step,num_order,ep_start

# Num epochs == num EPs.
class GreedyAgentRLModel(pl.LightningModule):
    environment_parameters: EnvironmentParameters
    agent: Agent

    DEBUG=False

    def __init__(self,
                 agent,
                 env: ShippingFacilityEnvironment,
                 experiment_name="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.env = env
        self.physical_network = self.env.environment_parameters.network
        self.experiment_name = experiment_name

        #Running values
        self.state = self.env.reset()
        self.done = False
        self.reward = 0
        self.episode_counter = 0
        self.episode_reward = 0.0
        self.actions = []
        self.episode_rewards = []
        self.info = {}
        self.episodes_info = []

        # debug var for env reset
        self.was_reset=True

    def forward(self, *args, **kwargs):
        pass # do nothing related to NNs.

    def training_step(self, step_info: Tuple[int, int, int],num_batch):
        step, order, ep_start = step_info

        if ep_start:
            logging.info(f"Starting episode {self.episode_counter}")
            if not self.was_reset:
                logging.error("ERROR!!! EXPECTED ENV TO BE RESET.")
            else:
                self.was_reset = False

        action = self.agent.get_action(self.state)

        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        next_state, reward, done, info = self.env.step(action)

        # print(f"Got reward {reward} done {done}")
        self.agent.train((self.state, action, next_state, reward, done))

        self.state = next_state
        self.episode_reward+=reward

        if done:
            # update the info to store the reports
            self.info = info

        # Render the current state of the environment
        self.env.render()
        self.actions.append(action)
        self.episode_rewards.append(reward)

        logging.info("Logging current steps metrics")

        shim = (torch.ones(2, 2, requires_grad=True)-1).sum() # a dummy operation to trick ptl

        result = pl.TrainResult(minimize=shim) # use the train result just for logging purposes.
        result.log("reward", reward)
        result.log("episode_reward", self.episode_reward)
        result.log("episodes", self.episode_counter)

        return result

    def training_epoch_end(self, outputs):

        logging.info(f"Finishing episode {self.episode_counter}")
        # Finished one episode, store reports
        logging.info("Finished episode, storing information")
        self.episodes_info.append(self.info)

        self._wandb_custom_metrics(self.info)

        self.episode_counter += 1

        self._reset_env_and_metrics()
        return outputs

    def _reset_env_and_metrics(self):
        logging.info(f"=========== starting episode {self.episode_counter} loop ===========")
        logging.debug("Initial environment: ")
        self.env.render()
        self.state = self.env.reset()
        self.done = False
        self.reward = 0
        self.episode_reward = 0.0
        self.actions = []
        self.episode_rewards = []
        self.info = {}
        self.was_reset = True # Making sure PTL is doing its job.

    def train_dataloader(self) -> DataLoader:
        '''
            This custom dataloader forces to run one step at a time (batching doesn't make sense here.)
            it's just a fancy iterator.
        '''
        return DataLoader(dataset=ShippingFacilityEpisodesDataset(
                num_steps=self.env.environment_parameters.num_steps,
                orders_per_day=self.env.environment_parameters.order_generator.orders_per_day,
            ),
            batch_size=1,
            shuffle=False
        )

    def _wandb_custom_metrics(self, info): #todo duplicate, refactor
        movement_detail_report = info['movement_detail_report']

        # Calculate big ms
        big_m_episode_count = movement_detail_report.is_big_m.astype(int).sum()

        # Calculate interplant transports
        transports = movement_detail_report[(movement_detail_report.movement_type == "Transportation")]

        total_interplants = transports.transportation_units.sum()

        incoming_interplants = transports.groupby("destination_name")['transportation_units'].sum().to_dict()

        # Calculate assignments per shipping point.
        deliveries = movement_detail_report[(movement_detail_report.movement_type == "Delivery")]
        deliveries_per_shipping_point_units = deliveries.groupby(['source_name'])['customer_units'].sum().to_dict()
        deliveries_per_shipping_point_orders = deliveries.drop_duplicates(["source_name","destination_name","source_time","destination_time"]).groupby("source_name").size().to_dict()
        mean_dcs_per_customer = deliveries.groupby(['destination_name'])['source_name'].nunique().reset_index().source_name.mean()


        # Per shipping point-customer?


        logging.info(f"Episode {self.episode_counter} had {big_m_episode_count} BigMs")

        wandb.log({
            "big_m_count": big_m_episode_count,
            "incoming_interplants": incoming_interplants,
            "total_interplants": total_interplants,
            "deliveries_per_shipping_point_units": deliveries_per_shipping_point_units,
            "deliveries_per_shipping_point_orders": deliveries_per_shipping_point_orders,
            "mean_dcs_per_customer": mean_dcs_per_customer,
        }, commit=False)

    def configure_optimizers(self):
        return [Adam([torch.ones(2,2,requires_grad=True)])]# shouldn't use it at all.

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        return

def main():
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
            "big_m_factor": 10000  # how many times the customer cost is the big m.

        },
        "hps": {
            "env": "shipping-v0", #openai env ID.
            "episode_length": 30, # todo isn't this an env thing?
            "max_episodes": 35,  # to do is this num episodes, is it being used?
            "batch_size": 30,
            "sync_rate": 2, # Rate to sync the target and learning network.
        },
        "seed":0,
        "agent": "best_fit"
        # "agent": "random_valid"
    }

    torch.manual_seed(config_dict['seed'])
    np.random.seed(config_dict['seed'])
    random.seed(config_dict['seed']) # not sure if actually used
    np.random.seed(config_dict['seed'])

    run = wandb.init(config=config_dict) # todo why not saving config???

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    experiment_name = f"gr_{config.agent}_few_warehouses"
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            # "debug"
            "experiment"
        ],
        log_model=False

    )

    wandb_logger.log_hyperparams(dict(config))

    environment_parameters = network_flow_env_builder.build_network_flow_env_parameters(
        environment_config,
        hparams['episode_length'],
        order_gen='biased'
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = agents.get_greedy_agent(env,config.agent)

    model = GreedyAgentRLModel(agent, env, experiment_name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=hparams['max_episodes'],
        early_stop_callback=False,
        val_check_interval=100,
        logger=wandb_logger,
        log_save_interval=1,
        row_log_interval=1, # the default of this may leave info behind.
        callbacks=[
            MyPrintingCallback(),
            ShippingFacilityEnvironmentStorageCallback(experiment_name, base="data/results/",
                                                       experiment_uploader=WandbDataUploader())
        ]
    )

    trainer.fit(model)


if __name__ == '__main__':
    # logging.root.level = logging.INFO
    logging.root.level = logging.DEBUG
    main()