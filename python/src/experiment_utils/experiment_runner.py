from envs.network_flow_env import (
    ActualOrderGenerator,
    NaiveInventoryGenerator,
    EnvironmentParameters,
    ShippingFacilityEnvironment,
    RandomAgent,
)
from gym.vector.utils import spaces

from network.PhysicalNetwork import PhysicalNetwork
import numpy as np


class ExperimentRunner:
    environment_parameters: EnvironmentParameters
    def __init__(self, order_generator, inventory_generator, agent, physical_network:PhysicalNetwork):
        self.order_generator = order_generator
        self.inventory_generator = inventory_generator
        self.agent = agent
        self.physical_network = physical_network
        self.environment_parameters = None


    def run_episode(self,num_steps):
        self.environment_parameters = EnvironmentParameters(
            self.physical_network, num_steps, self.order_generator, self.inventory_generator
        )

        env = ShippingFacilityEnvironment(self.environment_parameters)
        # agent = RandomAgent(env.action_space)

        obs = env.reset()
        reward = 0
        done = False
        print("=========== starting episode loop ===========")
        print("Initial environment: ")
        env.render()
        actions = []
        episode_rewards = []
        # demands_per_k = np.zeros((num_commodities,num_steps))
        # inventory_at_t = np.zeros((num_commodities,num_steps)) #todo llenar estos eventualmente
        while not done:
            action = self.agent.act(obs, reward, done)

            # print(f"Agent is taking action: {action}")
            # the agent observes the first state and chooses an action
            # environment steps with the agent's action and returns new state and reward
            obs, reward, done, info = env.step(action)
            # print(f"Got reward {reward} done {done}")

            # Render the current state of the environment
            env.render()
            actions.append(action)
            episode_rewards.append(reward)

            if done:
                print("===========Environment says we are DONE ===========")

        return actions, episode_rewards

    def run_episodes(self,num_steps,num_episodes,orders_per_day,experiment_name):
        total_rewards = []
        average_rewards = []
        total_actions = np.zeros(num_steps * orders_per_day)
        elapsed = []
        for i in range(num_episodes):
            start_time = time.process_time()
            actions, episode_rewards = self.run_episode(num_steps)
            end_time = time.process_time()

            total_rewards.append(sum(episode_rewards))
            average_rewards.append(np.mean(episode_rewards))
            elapsed.append(end_time - start_time)
            total_actions += np.array(actions)

        # Create datasets
        rewards_df = pd.DataFrame(data={
            'experiment_name': [experiment_name] * num_episodes,
            'episode': list(range(num_episodes)),
            'total_reward': total_rewards,
            'average_reward': average_rewards,
            'elapsed': elapsed
        })
        actions_df = pd.DataFrame(total_actions)

        base = f"data/results/{experiment_name}"
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists("data/results"):
            os.mkdir("data/results")
        if not os.path.exists(base):
            os.mkdir(base)
        rewards_df.to_csv(base + "/rewards.csv")
        actions_df.to_csv(base + "/actions.csv")
        print("done")


#TODO create a different version of this to use another agent.
def create_random_experiment_runner(num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day
    ):
    physical_network = PhysicalNetwork(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = NaiveInventoryGenerator()
    agent = RandomAgent(spaces.Discrete(
            num_dcs
    ))
    return ExperimentRunner(order_generator,generator,agent,physical_network)


class AlwaysFirstAgent(object):
    """The world's DUMBEST agent!"""

    def act(self, observation, reward, done):
        return 0


def create_always_first_dc_agent(num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day
    ):
    physical_network = PhysicalNetwork(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = NaiveInventoryGenerator()
    agent = AlwaysFirstAgent()
    return ExperimentRunner(order_generator,generator,agent,physical_network)



def run_with_params(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps

):
    physical_network = PhysicalNetwork(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    # order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = NaiveInventoryGenerator()
    environment_parameters = EnvironmentParameters(
        physical_network, num_steps, order_generator, generator
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    reward = 0
    done = False
    print("=========== starting episode loop ===========")
    print("Initial environment: ")
    env.render()
    actions = []
    episode_rewards = []
    #demands_per_k = np.zeros((num_commodities,num_steps))
    #inventory_at_t = np.zeros((num_commodities,num_steps)) #todo llenar estos eventualmente
    while not done:
        action = agent.act(obs, reward, done)

        # print(f"Agent is taking action: {action}")
        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        obs, reward, done, info = env.step(action)
        # print(f"Got reward {reward} done {done}")

        # Render the current state of the environment
        env.render()
        actions.append(action)
        episode_rewards.append(reward)

        if done:
            print("===========Environment says we are DONE ===========")

    return actions, episode_rewards

import os
import pandas as pd
import time
def run_episodes(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
        num_episodes,
        experiment_name
):
    total_rewards = []
    average_rewards = []
    total_actions = np.zeros(num_steps*orders_per_day)
    elapsed = []
    for i in range(num_episodes):
        start_time = time.process_time()
        actions, episode_rewards = run_with_params(num_dcs,
            num_customers,
            dcs_per_customer,
            demand_mean,
            demand_var,
            num_commodities,
            orders_per_day,
            num_steps
        )
        end_time = time.process_time()

        total_rewards.append(sum(episode_rewards))
        average_rewards.append(np.mean(episode_rewards))
        elapsed.append(end_time-start_time)
        total_actions += np.array(actions)

    #Create datasets
    rewards_df = pd.DataFrame(data={
        'experiment_name': [experiment_name] * num_episodes,
        'episode': list(range(num_episodes)),
        'total_reward': total_rewards,
        'average_reward': average_rewards,
        'elapsed': elapsed
    })
    actions_df = pd.DataFrame(total_actions)

    base = f"data/results/{experiment_name}"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/results"):
        os.mkdir("data/results")
    if not os.path.exists(base):
        os.mkdir(base)
    rewards_df.to_csv(base+"/rewards.csv")
    actions_df.to_csv(base+"/actions.csv")
    print("done")


