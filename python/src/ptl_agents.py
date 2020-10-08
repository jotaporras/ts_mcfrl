import os
import random
import shutil
from collections import deque

import torch
import wandb
from envs import ShippingFacilityEnvironment
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


# Named tuple for storing experience steps gathered in training
from experiment_utils import report_generator
from network.PhysicalNetwork import PhysicalNetwork

Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class DQN(nn.Module):
    """
    Simple MLP network.
    From

    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """
    Base Agent class handeling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state['state_vector'].reshape(-1)])

            if device not in ['cpu']:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment

        new_state, reward, done, info = self.env.step(action)

        exp = Experience(self.state['state_vector'].reshape(-1), action, reward, done, new_state['state_vector'].reshape(-1))

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done, info


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, hparams: argparse.Namespace,environment_parameters:EnvironmentParameters) -> None:
        super().__init__()
        self.hparams = hparams

        self.env = ShippingFacilityEnvironment(environment_parameters)
        obs_size = self.env.observation_space.shape[1] # todo changed from the default [0] in the example, maybe it was a standard.
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)# todo change
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0

        # Initialize episode information for debugging.
        self.episodes_info = []
        self.episode_counter = 0

        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            _, done, info = self.agent.play_step(self.net, epsilon=1.0)
            if done:
                self.store_episode_info(info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      self.global_step + 1 / self.hparams.eps_last_frame)

        # step through environment with agent
        reward, done, info = self.agent.play_step(self.net, epsilon, device) # refill the buffer every time.
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if done:
            self.store_episode_info(info)

        result = pl.TrainResult(minimize=loss)

        logging.info("Logging current steps metrics")

        result.log("loss", loss)
        result.log("reward", reward)  # todo check if correct and if it fits.
        result.log("total_reward", self.total_reward)  # todo check if correct and if it fits.
        result.log("episode_reward", self.episode_reward)
        result.log("episodes", self.episode_counter)

        return result

    def store_episode_info(self, info):
        logging.info("Finished episode, storing information")
        self.episodes_info.append(info)
        self.episode_counter += 1
        self.episode_reward = 0

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_epoch_end(self,trainer,pl_module):
        print("epoch is done")

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')

class WandbDataUploader():
    def __init__(self, base="data/results"):
        self.base = base

    def upload(self, experiment_name):
        # copy base to wandb.
        shutil.copytree(os.path.join(self.base,experiment_name), os.path.join(wandb.run.dir,self.base,experiment_name))
        #wandb.save(os.path.join(wandb.run.dir,self.base,experiment_name))
        # wandb.save(os.path.join(self.base,experiment_name+"*"))

class ShippingFacilityEnvironmentStorageCallback(Callback):
    '''
        Stores the information objects into CSVs for debugging Environment and actions.
    '''
    def __init__(self, experiment_name,base,experiment_uploader:WandbDataUploader):
        self.experiment_name = experiment_name
        self.base = base
        self.experiment_uploader = experiment_uploader

    def on_train_end(self, trainer, pl_module):
        logging.info("Finished training, writing environment info objects") # todo aqui quede, see if this works and if it is sufficient reporting.
        report_generator.write_single_df_experiment_reports(pl_module.episodes_info, self.experiment_name,self.base)
        report_generator.write_generate_physical_network_valid_dcs(pl_module.env.environment_parameters.network, self.experiment_name,self.base)

        self.experiment_uploader.upload(self.experiment_name)


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
        },
        "hps": {
            "env": "shipping-v0", #openai env ID.
            "replay_size": 150,
            "warm_start_steps": 150, # apparently has to be smaller than batch size
            "max_epochs": 5000, # to do is this num episodes, is it being used?
            "episode_length": 30, # todo isn't this an env thing?
            "batch_size": 30,
            "gamma": 0.99,
            "eps_end": 1.0, #todo consider keeping constant to start.
            "eps_start": 0.99, #todo consider keeping constant to start.
            "eps_last_frame": 1000, # todo maybe drop
            "sync_rate": 2, # Rate to sync the target and learning network.
            "lr": 1e-2,
        }
    }
    # Debug dict
    # config_dict = {
    #     "env": {
    #         "num_dcs": 3,
    #         "num_customers": 10,
    #         "num_commodities": 5,
    #         "orders_per_day": 1,
    #         "dcs_per_customer": 2,
    #         "demand_mean": 500,
    #         "demand_var": 150,
    #         "num_steps": 10,  # steps per episode
    #     },
    #     "hps": {
    #         "env": "shipping-v0", #openai env ID.
    #         "replay_size": 30,
    #         "warm_start_steps": 30, # apparently has to be smaller than batch size
    #         "max_epochs": 20, # to do is this num episodes, is it being used?
    #         "episode_length": 30, # todo isn't this an env thing?
    #         "batch_size": 30,
    #         "gamma": 0.99,
    #         "eps_end": 1.0, #todo consider keeping constant to start.
    #         "eps_start": 0.99, #todo consider keeping constant to start.
    #         "eps_last_frame": 1000, # todo maybe drop
    #         "sync_rate": 2, # Rate to sync the target and learning network.
    #         "lr": 1e-2,
    #     }
    # }

    run = wandb.init(config=config_dict)

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    experiment_name = "dqn_few_warehouses_v4"
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            #"debug"
            "experiment"
        ],
        log_model=True

    )


    wandb_logger.log_hyperparams(dict(config))

    physical_network = PhysicalNetwork(
        num_dcs = environment_config['num_dcs'],
        num_customers = environment_config['num_customers'],
        dcs_per_customer = environment_config['dcs_per_customer'],
        demand_mean = environment_config['demand_mean'],
        demand_var = environment_config['demand_var'],
        num_commodities = environment_config['num_commodities'],
    )
    order_generator = ActualOrderGenerator(physical_network, environment_config['orders_per_day'])
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network,
        hparams['episode_length'],
        order_generator, generator
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

    trainer.fit(model)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0) # not sure if actually used
    np.random.seed(0)

    main()