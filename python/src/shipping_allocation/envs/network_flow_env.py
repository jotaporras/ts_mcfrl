from abc import ABC
from typing import List, Any

import gym


import typing
from gym import spaces
import random
import numpy as np

from experiment_utils import network_flow_k_optimizer
from network.Network import Network
from orders.Order import Order

# Abstract classes. BARNUM, you have to implement these
class OrderGenerator(ABC):
    # Generates a set of orders for the next timestep.
    def generate_orders(self) -> List[Order]:
        pass


class InventoryGenerator(ABC):
    # Generates new inventory and distributes it somehow to keep the network balanced for the selected orders.
    # Returns a numpy array of shape (num_dcs,num_commodities) representing how much extra inventory is going to appear.
    def generate_new_inventory(
        self, network: Network, open_orders: List[Order]
    ):  # todo add type when it works.
        pass


# All the immutable things that affect environment behavior, maybe needs more parameters?
class EnvironmentParameters:
    def __init__(
        self,
        network: Network,
        num_episodes: int,
        order_generator: OrderGenerator,
        inventory_generator: InventoryGenerator,
    ):
        self.network = network
        self.num_episodes = num_episodes
        self.order_generator = order_generator
        self.inventory_generator = inventory_generator


class ShippingFacilityEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    open_orders: List[Order]
    metadata = {"render.modes": ["human"]}

    def __init__(self, environment_parameters: EnvironmentParameters):
        super(ShippingFacilityEnvironment, self).__init__()
        self.environment_parameters = environment_parameters
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []  # Orders not yet decided upon.
        self.current_state = {}
        self.current_t = 0
        self.action_space = spaces.Discrete(
            self.environment_parameters.network.num_dcs
        )  # The action space is choosing a DC for the current order.
        self.inventory = np.zeros(
            (
                self.environment_parameters.network.num_dcs,
                self.environment_parameters.network.num_commodities # TODO this field is missing.
            )
        )  # Matrix of inventory for each dc-k.

        print("Calling init on the ShippingFacilityEnvironment")

    # Taking a step forward after the agent selects an action for the current state.
    def step(self, action):
        # Choose the shipping point for the selected order and fix the order.

        self.open_orders[0].initialPoint = action#self.environment_parameters.network.dcs[action] #TODO talk to barnum about this parameter type.
        self.fixed_orders = self.fixed_orders + [self.open_orders[0]]

        # Remove it from open orders
        self.open_orders = self.open_orders[1:]

        cost = self._run_simulation()

        reward = cost * -1

        # update timestep and generate new orders if needed
        self.current_state = self._next_observation()

        # Done when the number of timestep generations is the number of episodes.
        done = self.current_t == self.environment_parameters.num_episodes + 1

        # print(f"Stepping with action {action}")
        # obs = random.randint(0, 10)
        # reward = random.randint(0, 100)
        # done = np.random.choice([True, False])
        return self.current_state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        print("Reseting environment")
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []
        self.current_t = 1
        self.current_state = self._next_observation()

        return self.current_state

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        if mode == "human":
            print(f"fixed_orders ({len(self.fixed_orders)})",self.fixed_orders)
            print(f"Demand fixed orders: {sum(map(lambda o:o.capacity,self.fixed_orders))}") #TODO Do for all commodities
            print(f"open_orders ({len(self.open_orders)})",self.open_orders)
            print(f"Demand open orders: {sum(map(lambda o:o.capacity,self.open_orders))}") #TODO Do for all commodities
            print("inventory\n",self.inventory)
            print("rendering")


    def _next_observation(self):
        if len(self.open_orders) == 0:  # Create new orders if necessary
            new_orders = self._generate_orders()
            self.open_orders = self.open_orders + new_orders
            self.current_t += 1
        self.inventory = self._generate_updated_inventory()
        return {
            "physical_network": self.environment_parameters.network,
            "inventory": self.inventory,
            "open": self.open_orders,
            "fixed": self.fixed_orders,
            "current_t": self.current_t,
        }

    def _generate_orders(self) -> typing.List[Order]:
        print("Calling order generator")
        return self.environment_parameters.order_generator.generate_orders()

    def _generate_updated_inventory(self):
        new_inventory = self.environment_parameters.inventory_generator.generate_new_inventory(self.environment_parameters.network, self.open_orders) #must keep shape
        return self.inventory + new_inventory

    def _run_simulation(self) -> float:
        return network_flow_k_optimizer.optimize(self.current_state)


# Naive implementations of inventory and order generators to illustrate.
class NaiveOrderGenerator(OrderGenerator):
    default_delivery_time = 1
    def __init__(self, num_dcs, num_customers, orders_per_day):
        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.orders_per_day = orders_per_day

    def generate_orders(self):  # TODO: needs a list of commodities, also needs the
        customer = "c_" + str(np.random.choice(np.arange(self.num_customers)))
        dc = "dc_" + str(np.random.choice(np.arange(self.num_dcs)))
        demand = random.randint(0, 50)
        return [Order(demand, dc, customer, self.default_delivery_time) for it in range(self.orders_per_day)]


class NaiveInventoryGenerator(InventoryGenerator):
    def generate_new_inventory(self, network: Network, open_orders: List[Order]):
        total_inventory = sum(
            map(lambda o: o.capacity, open_orders)
        )  # TODO rename and do for many commmodities.
        even = total_inventory // network.num_dcs
        dc_inv = np.array([even] * network.num_dcs).reshape(
            -1, 1
        )  # To keep the (dc,product) shape.
        if total_inventory // network.num_dcs != total_inventory / network.num_dcs:
            dc_inv[0, 0] = dc_inv[0, 0] + 1
        return dc_inv

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == "__main__":
    num_dcs=5
    num_customers=5
    num_episodes = 10
    orders_per_day=2
    dcs_per_customer=2
    physical_network = Network(num_dcs, num_customers, dcs_per_customer)
    order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
    generator = NaiveInventoryGenerator()
    environment_parameters = EnvironmentParameters(physical_network, num_episodes, order_generator, generator)

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    reward = 0
    done=False
    print("=========== starting episode loop ===========")
    print("Initial environment: ")
    env.render()
    while not done:
        action = agent.act(obs, reward, done)
        print(f"Agent is taking action: {action}")
        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        obs, reward, done, info = env.step(action)
        print(f"Got reward {reward} done {done}")

        # Render the current state of the environment
        env.render()

        if done:
            print("===========Environment says we are DONE ===========")