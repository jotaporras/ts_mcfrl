from abc import ABC
from typing import List, Any

import typing
import random
import numpy as np
import copy

# Environment and agent
import gym
import scipy
import scipy.stats
from gym import spaces
from gym import wrappers
import tensorflow.compat.v1 as tf

from locations import Orders
import logging

tf.disable_v2_behavior()

#Custome
from experiment_utils import network_flow_k_optimizer, report_generator
from network.PhysicalNetwork import PhysicalNetwork
from locations.Order import Order
DEBUG=False

# Abstract classes. BARNUM, you have to implement these
class OrderGenerator(ABC):
    # Generates a set of locations for the next timestep.
    def generate_orders(self,current_t:int) -> List[Order]:
        pass


class InventoryGenerator(ABC):
    # Generates new inventory and distributes it somehow to keep the network balanced for the selected locations.
    # Returns a numpy array of shape (num_dcs,num_commodities) representing how much extra inventory is going to appear.
    def generate_new_inventory(
        self, network: PhysicalNetwork, open_orders: List[Order]
    ):  # todo add type when it works.
        pass


# All the immutable things that affect environment behavior, maybe needs more parameters?
class EnvironmentParameters:
    def __init__(self, network: PhysicalNetwork, order_generator: OrderGenerator,
                 inventory_generator: InventoryGenerator, num_steps: int):
        self.network = network
        self.num_steps = num_steps
        self.order_generator = order_generator
        self.inventory_generator = inventory_generator


class ShippingFacilityEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""

    open_orders: List[Order]
    metadata = {"render.modes": ["human"]}

    def __init__(self, environment_parameters: EnvironmentParameters):
        super(ShippingFacilityEnvironment, self).__init__()

        self.environment_parameters = environment_parameters
        self.all_movements_history = []
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
                self.environment_parameters.network.num_commodities,
            )
        )  # Matrix of inventory for each dc-k.
        self.transports_acc = np.zeros(self.inventory.shape)

        #=====OBSERVATION SPEC=======
        # Current observation spec:
        # inventory per dc plus one K size vector of the current order
        # + one K size vector of other demand in horizon
        # + 4 metadata neurons
        dcs = self.environment_parameters.network.num_dcs
        commodities = self.environment_parameters.network.num_commodities
        shape = (1, dcs * commodities + 2*commodities + 4)
        self.observation_space = spaces.Box(0, 1000000, shape=shape)
        # =====OBSERVATION SPEC=======

        self.last_cost=0

        # Debug vars
        self.approx_transport_mvmt_list = []
        self.total_costs = []
        self.total_rewards = []
        self.info = {}

        logging.info("Calling init on the ShippingFacilityEnvironment")

    # Taking a step forward after the agent selects an action for the current state.
    def step(self, action):
        # Choose the shipping point for the selected order and fix the order.
        if DEBUG:
            logging.info("\n=============================================")
            logging.info(f"===============> STARTING ENVIRONMENT STEP {self.current_t}")
            logging.info("=============================================")
            logging.info(f"Received action {action}")
            logging.info("Pre env.step render:")
            #logging.info("Current state: ",self.current_state)
            self.render()

        #"Setting shipping point",action," for order ",self.open_orders[0])
        new_shipping_point = self.environment_parameters.network.dcs[action]
        self.open_orders[0].shipping_point = new_shipping_point
        #logging.info("Order after seting action: ",self.open_orders[0])
        self.fixed_orders = self.fixed_orders + [self.open_orders[0]]
        self.current_state['fixed'] = self.fixed_orders #TODO cleanup state update

        # Remove it from open locations
        self.open_orders = self.open_orders[1:]
        self.current_state['open'] = self.open_orders #TODO cleanup state update

        cost, transports, all_movements = self._run_simulation()
        reward = (self.last_cost - cost)*-1.0 # todo: will first step be an issue? because no last cost means big positive.
        #reward = cost * -1 # old reward function.
        self.last_cost = cost

        # Adding approximate transport cost by taking transport matrices where transports are greater than zero. Assumes Customer transport == DC transport cost.
        # Only append movements after open orders have been depleted, meaning once per day.
        if len(self.open_orders) == 0:
            logging.info("Finished the day, adding all movements")
            self.all_movements_history.append(all_movements)
        else:
            logging.info("Still are open orders left, no report added")
            logging.info(f"Number of orders left {len(self.open_orders)}")
        self.approximate_transport_cost = transports[np.where(transports > 0)].sum() * self.environment_parameters.network.default_customer_transport_cost
        self.total_costs.append(cost)
        self.total_rewards.append(reward)
        self.approx_transport_mvmt_list.append(self.approximate_transport_cost)
        #logging.info("self.approx_transport_cost", self.approx_transport_cost)
        #logging.info("Total transports (customer+dc) transports")

        self.transports_acc = transports



        # update timestep and generate new locations if needed
        self.current_state = self._next_observation()

        # Done when the number of timestep generations is the number of episodes.
        done = self.current_t == self.environment_parameters.num_steps + 1

        if done:
            # Appending final values to info object.
            final_ords:List[Order] = self.current_state['fixed']

            movement_detail_report = report_generator.generate_movement_detail_report(self.all_movements_history,self.environment_parameters.network.big_m_cost)
            summary_movement_report = report_generator.generate_summary_movement_report(movement_detail_report)

            served_demand = sum([sum(o.demand) for o in final_ords])
            approximate_to_customer_cost = served_demand*self.environment_parameters.network.default_customer_transport_cost
            self.info = {
                'final_orders': final_ords, # The orders, with their final shipping point destinations.
                'total_costs': self.total_costs, # Total costs per stepwise optimization
                'approximate_transport_movement_list': self.total_costs, # Total costs per stepwise optimization
                'approximate_to_customer_cost': approximate_to_customer_cost, # Approximate cost of shipping to customers: total demand multiplied by the default customer transport cost. If the cost is different, this is worthless.
                'movement_detail_report': movement_detail_report, # DataFrame with all the movements that were made.
                'summary_movement_report': summary_movement_report  # DataFrame with movements summary per day.
            }
            # logging.info("==== Copy and paste this into a notebook ====")
            # logging.info("Total costs per stepwise optimization", self.total_costs)
            # logging.info("Total cost list associated with all transport movements", self.approx_transport_mvmt_list) #approximate because they're intermixed.
            # logging.info("Removing approx to customer cost", sum(self.approx_transport_mvmt_list)-approximate_to_customer_cost)
        else:
            self.info={}# not done yet. #to do consider yielding this on every step for rendering purposes.
        # logging.info(f"Stepping with action {action}")
        # obs = random.randint(0, 10)
        # reward = random.randint(0, 100)
        # done = np.random.choice([True, False])
        return copy.copy(self.current_state), reward, done, self.info

    # def observation_space(self):
    #     dcs = self.environment_parameters.network.num_dcs
    #     commodities = self.environment_parameters.network.num_commodities
    #     shape = (dcs * commodities+num_commodities, 1)
    #     return spaces.Box(0,1000000,shape=shape)

    def generate_final_statistics(self):
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        #logging.info("Physical network for new env: ")
        #logging.info(self.environment_parameters.network)
        #logging.info("Reseting environment")
        self.inventory = np.zeros(
            (
                self.environment_parameters.network.num_dcs,
                self.environment_parameters.network.num_commodities,
            )
        )  # Matrix of inventory for each dc-k.
        self.all_movements_history = []
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []

        self.current_t = 0
        self.current_state = self._next_observation()

        #debug var
        self.approx_transport_mvmt_list = []
        self.total_costs = []
        self.total_rewards = []

        return copy.copy(self.current_state)



    def render(self, mode="human", close=False):
        pass #todo refactor this, it's too much noise
        # Render the environment to the screen
        # if mode == "human" and DEBUG:
        #     logging.info("\n\n======== RENDERING ======")
        #     logging.info("Current t",self.current_t)
        #     logging.info(f"fixed_orders ({len(self.fixed_orders)})", self.fixed_orders)
        #     logging.info(
        #         f"Demand fixed orders: {sum(map(lambda o:o.demand, self.fixed_orders))}"
        #     )  # TODO Do for all commodities
        #     logging.info(f"open_orders ({len(self.open_orders)})", self.open_orders)
        #     logging.info(
        #         f"Demand open orders: {sum(map(lambda o:o.demand, self.open_orders))}"
        #     )  # TODO Do for all commodities
        #     logging.info("inventory\n", self.inventory)
        #     logging.info("Current State:")
        #     self._render_state()
        #     logging.info("======== END RENDERING ======\n\n")

    def _render_state(self):
        if DEBUG:
            logging.info("Rendering mutable part of the state")
            logging.info("fixed: ",self.current_state['fixed'])
            logging.info("open: ",self.current_state['open'])
            logging.info("inventory: ", self.current_state['inventory'])
            logging.info("current_t: ", self.current_state['current_t'])

    def _next_observation(self):
        if len(self.open_orders) == 0:  # Create new locations if necessary
            #if self.current_t != 0: #Dont update the T if it's the start of the run/ #TODO VALIDATE THIS MIGHT BE AN ISSUE!!!!!!!


            consumed_inventory = self._calculate_consumed_inventory()
            self.current_t += 1
            new_orders = self._generate_orders()
            self.open_orders = self.open_orders + new_orders

            #logging.info("Updating inventory with orders")
            #logging.info("Before update: ")
            #logging.info(self.inventory)

            self.inventory = self._generate_updated_inventory(consumed_inventory)

            #logging.info("inventory after orders before transports")
            #logging.info(self.inventory)

            if (self.transports_acc > 0).any():
                # logging.info("Applying transports!!! Transports:***")
                # logging.info(self.transports_acc)
                self.inventory += self.transports_acc
                # logging.info("New inventory after transports")
                # logging.info(self.inventory)
                # logging.info("setting all to zero again")
                self.transports_acc[:, :] = 0
                # logging.info(self.transports_acc)



            if (self.inventory<0).any():
                logging.info("THIS SHOULDNT HAPPEN!!!!! NEGATIVE INVENTORY")
                logging.info(self.inventory)
                raise Exception("THIS SHOULDNT HAPPEN!!!!! NEGATIVE INVENTORY")
        # else:
        #     self.inventory = self._generate_updated_inventory(0)
        generated_state = {
            "physical_network": self.environment_parameters.network,
            "inventory": self.inventory.copy(),
            "open": [copy.deepcopy(o) for o in self.open_orders],
            "fixed": [copy.deepcopy(o) for o in self.fixed_orders],
            "current_t": self.current_t
        }
        generated_state['state_vector'] = self.convert_state_to_vector(generated_state) # todo refactor to something less nasty.
        return generated_state

    def convert_state_to_vector(self, state): # copied from first dqn agent
        # Inventory stack
        inventory = state['inventory']
        stacked_inventory = inventory.reshape(-1,1)

        # latest open order demand
        latest_open_order = state['open'][0]
        reshaped_demand = latest_open_order.demand.reshape(-1,1)*-1

        # Calculating stacked demand in horizon.
        fixed_orders: List[Order] = state['fixed']
        current_t = state['current_t']
        network: PhysicalNetwork = state['physical_network']
        horizon = current_t + network.planning_horizon - 1
        stacked_demand_in_horizon = Orders.summarize_order_demand(fixed_orders, current_t, horizon, reshaped_demand.shape)*-1

        #4 extra metadata neurons.
        ship_id = latest_open_order.shipping_point.node_id
        customer_id = latest_open_order.customer.node_id
        current_t = state['current_t']
        delivery_t = latest_open_order.due_timestep
        metadata = np.array([[ship_id, customer_id, current_t, delivery_t]]).transpose()

        # State vector
        state_vector = np.concatenate([stacked_inventory, reshaped_demand, stacked_demand_in_horizon, metadata])

        return state_vector.transpose() #np.array((1,num_dcs*num_commodities + num_commodities))

    def _generate_orders(self) -> typing.List[Order]:
        #logging.info(f"Calling order generator for t={self.current_t}")
        return self.environment_parameters.order_generator.generate_orders(self.current_t)

    def _generate_updated_inventory(self,consumed_inventory):
        new_inventory = self.environment_parameters.inventory_generator.generate_new_inventory(
            self.environment_parameters.network, self.open_orders
        )  # must keep shape
        return self.inventory + new_inventory - consumed_inventory

    def _calculate_consumed_inventory(self):
        #logging.info("Calculating consumed inventory")
        consumed = np.zeros(self.inventory.shape)
        for order in self.fixed_orders:
            if order.due_timestep == self.current_t:
                #logging.info("Order",order.name,"is getting consumed on timelapse ",self.current_t," from ",order.shipping_point)
                consumed[order.shipping_point.node_id,:] += order.demand
        # logging.info("Consumed inventory: ")
        # logging.info(consumed)
        return consumed

    def _run_simulation(self) -> float:
        return network_flow_k_optimizer.optimize(self.current_state) #TODO aqui quede implement the K optimizer given the order gen.


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
        return [
            Order(demand, dc, customer, self.default_delivery_time)
            for it in range(self.orders_per_day)
        ]


class ActualOrderGenerator(OrderGenerator):
    network: PhysicalNetwork
    orders_per_day: int

    def __init__(self, network: PhysicalNetwork, orders_per_day):
        self.network = network
        self.orders_per_day = orders_per_day

    def generate_orders(self, current_t) -> List[Order]:
        return self._generate_orders(self.orders_per_day, current_t)

    def _generate_orders(self, orders_per_day: int, current_t):  # TODO test and validate.
        # Choose customers to generate orders with OUT replacement, orders per day must be <= customers
        chosen_customers = np.random.choice(np.arange(self.network.num_customers), size=orders_per_day, replace=False)
        order_means = self.network.customer_means[chosen_customers]

        demand = np.floor(np.random.multivariate_normal(order_means, np.eye(orders_per_day) * self.network.demand_var,
                                                        size=self.network.num_commodities))  # shape (num_commodities,num_orders)
        if (demand < 0).any():
            logging.info("Customer means that caused negatives")
            logging.info(order_means)
            # raise Exception("Generated a negative order")
            demand = np.abs(demand)
        # Create order objects
        orders = []
        for ci in range(len(chosen_customers)):
            order_demand_vector = demand[:, ci]
            _chosen_customer = chosen_customers[ci]
            customer_node = self.network.customers[_chosen_customer]
            chosen_initial_point = np.random.choice(np.argwhere(self.network.dcs_per_customer_array[ci, :]).reshape(-1))
            initial_point_physical_node = self.network.dcs[chosen_initial_point]
            time = current_t + self.network.planning_horizon - 1  # Orders appear on the edge of PH.
            orders.append(Order(order_demand_vector, initial_point_physical_node, customer_node, time,
                                name=f"oc_{customer_node.node_id}:{time}"))
        return orders

class BiasedOrderGenerator(OrderGenerator):
    network: PhysicalNetwork
    orders_per_day: int
    customer_means: np.array

    def __init__(self, network: PhysicalNetwork, orders_per_day):
        self.network = network
        self.orders_per_day = orders_per_day
        self.customer_covariances = self._generate_customer_covariances() #shape:(C,K,K)
        self.customer_means = self._generate_customer_means()

    def _generate_customer_covariances(self):
        K = self.network.num_commodities
        num_customers = self.network.num_customers
        return scipy.stats.invwishart(K,np.ones(K)).rvs(size=num_customers)

    def _generate_customer_means(self):
        #total_demand_mean = self.network.demand_mean * self.network.num_customers * self.network.num_commodities
        return np.random.poisson(self.network.demand_mean/self.network.num_commodities,size=self.network.num_commodities)
        # return np.floor(
        #     np.random.dirichlet(self.network.num_commodities / np.arange(1, self.network.num_commodities + 1),
        #                         size=1) * total_demand_mean).reshape(-1) + self.network.demand_mean # shape (commodities)

    def generate_orders(self, current_t) -> List[Order]:
        # todo params
        chosen_customers = np.random.choice(np.arange(self.network.num_customers), size=self.orders_per_day, replace=False)
        order_means = self.network.customer_means[chosen_customers] # getting the means from the network but the covariances from here for legacy reasons.
        K = self.network.num_commodities

        ####

        # Generating covariance matrix with inverse Wishart distribution. What does that parameter do?
        covar = scipy.stats.invwishart(K,np.ones(K)).rvs(size=1)

        orders = []
        for ci in range(len(chosen_customers)):
            means = self.customer_means
            covar = self.customer_covariances[ci,:,:]

            # Sampling X from a multivariate normal with the covariance from Wishart.
            multivariate_normal_x = np.random.multivariate_normal(np.zeros(means.shape), covar, size=1)

            # Extract the probability density of the sampled values. Is the sqrt(diag(covar)) arbitrary?
            px = scipy.stats.norm(0, np.sqrt(np.diagonal(covar))).cdf(multivariate_normal_x)

            # Take those quantiles and plug them into a geometric. This is going to skew the data and project it into the range that we want starting at 0.
            # qgeom(x,prob). X is a vector of quantiles of the probability of failures in a Bernoulli (shape K). Second param is probabilities.  Why pz(1-pz)?? Something related to MLE?
            pz = 1 / means
            order_demand = scipy.stats.geom(p=pz * (1 - pz)).ppf(px).flatten()

            _chosen_customer = chosen_customers[ci]
            customer_node = self.network.customers[_chosen_customer]
            chosen_initial_point = np.random.choice(np.argwhere(self.network.dcs_per_customer_array[ci, :]).reshape(-1))
            initial_point_physical_node = self.network.dcs[chosen_initial_point]
            time = current_t + self.network.planning_horizon - 1  # Orders appear on the edge of PH.

            orders.append(Order(order_demand, initial_point_physical_node, customer_node, time,
                                name=f"oc_{customer_node.node_id}:{time}"))
        return orders



class NaiveInventoryGenerator(InventoryGenerator):
    def generate_new_inventory(
        self, network: PhysicalNetwork, open_orders: List[Order]
    ):
        #logging.info("==> inventory generator")
        total_inventory = sum(
            map(lambda o: o.demand, open_orders)
        )  # TODO rename and do for many commmodities.
        even = total_inventory // network.num_dcs
        dc_inv = np.array([even] * network.num_dcs).reshape(
            network.num_dcs,-1
        )  # To keep the (dc,product) shape. #todo validate with multiple commodities
        # logging.info("Demand", total_inventory)
        # logging.info("Pre level dc_inv")
        # logging.info(dc_inv)
        # logging.info("Total new inv",np.sum(dc_inv))
        imbalance = total_inventory - np.sum(dc_inv,axis=0)
        #if total_inventory // network.num_dcs != total_inventory / network.num_dcs:
        dc_inv[0, :] = dc_inv[0, :] + imbalance
        # logging.info("Rebalanced dc inv",dc_inv)
        # logging.info("Rebalanced sum",np.sum(dc_inv))
        if (np.sum(dc_inv,axis=0) != total_inventory).any():
            raise Exception("np.sum(dc_inv) != total_inventory")
        return dc_inv

class DirichletInventoryGenerator(InventoryGenerator):

    def __init__(self,network: PhysicalNetwork):
        num_dcs = network.num_dcs
        num_commodities = network.num_commodities
        self.alpha = np.random.permutation(num_dcs/np.arange(1,num_dcs+1)) # trying to make it skewed.
        self.inventory_generation_distribution = np.random.dirichlet(self.alpha, num_commodities)  # (num_dc,num_k) of dc distribution of inventory.

    def generate_new_inventory(
        self, network: PhysicalNetwork, open_orders: List[Order]
    ):
        #logging.info("==> inventory generator")
        total_inventory = sum(
            map(lambda o: o.demand, open_orders)
        )  # TODO rename and do for many commmodities.
        #even = total_inventory // network.num_dcs
        inventory_distribution = self.inventory_generation_distribution

        supply_per_dc = np.floor(total_inventory.reshape(-1, 1) * inventory_distribution)
        imbalance = total_inventory - np.sum(supply_per_dc, axis=1)
        supply_per_dc[:, 0] = supply_per_dc[:, 0] + imbalance

        # logging.info("Demand", total_inventory)
        # logging.info("Pre level dc_inv")
        # logging.info(dc_inv)
        # logging.info("Total new inv",np.sum(dc_inv))
        #if total_inventory // network.num_dcs != total_inventory / network.num_dcs:
        # logging.info("Rebalanced dc inv",dc_inv)
        # logging.info("Rebalanced sum",np.sum(dc_inv))
        if not np.isclose(np.sum(np.sum(supply_per_dc, axis=1) - total_inventory), 0.0):
            raise RuntimeError("Demand was not correctly balanced")
        return supply_per_dc.transpose()




# if __name__ == "__main__":
#     num_dcs = 2
#     num_customers = 1
#     num_commodities = 3
#     orders_per_day = 1
#     dcs_per_customer = 1
#     demand_mean = 100
#     demand_var = 20
#
#     num_episodes = 5
#
#     physical_network = PhysicalNetwork(num_dcs, num_customers, dcs_per_customer,demand_mean,demand_var,num_commodities)
#     # order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
#     order_generator = ActualOrderGenerator(physical_network, orders_per_day)
#     generator = NaiveInventoryGenerator()
#     environment_parameters = EnvironmentParameters(
#         physical_network, num_episodes, order_generator, generator
#     )
#
#     env = ShippingFacilityEnvironment(environment_parameters)
#     agent = QNAgent(env)
#
#     state = env.reset()
#     reward = 0
#     done = False
#     logging.info("=========== starting episode loop ===========")
#     logging.info("Initial environment: ")
#     env.render()
#     while not done:
#         action = agent.get_action((state, reward))
#         logging.info(f"Agent is taking action: {action}")
#         # the agent observes the first state and chooses an action
#         # environment steps with the agent's action and returns new state and reward
#         next_state, reward, done, info = env.step(action)
#         logging.info(f"Got reward {reward} done {done}")
#
#         agent.train((state,action,next_state,reward,done))
#
#         state = next_state
#         # Render the current state of the environment
#         env.render()
#
#         if done:
#             logging.info("===========Environment says we are DONE ===========")



#TODO aqui quede, works on first step pero no tengo la ventana de tiempo lista todavía, además está faltando un arco para la orden vieja.
# EJ:
# mcf.SetNodeSupply(0,int(114.0))
# mcf.SetNodeSupply(1,int(0))
# mcf.SetNodeSupply(2,int(0))
# mcf.SetNodeSupply(3,int(114.0))
# mcf.SetNodeSupply(4,int(0))
# mcf.SetNodeSupply(5,int(0))
# mcf.SetNodeSupply(6,int(-114.0))
# mcf.SetNodeSupply(7,int(-114.0))
# mcf.AddArcWithCapacityAndUnitCost(0, 1, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(1, 2, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(3, 4, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(4, 5, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(0, 3, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(1, 4, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(2, 5, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(3, 0, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(4, 1, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(5, 2, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(5, 7, 9000000, 10)
# Running optimization
#TODO : crear sliding window del TEN para que vaya avanzando y moviendo las conexiones.
#TODO: ALSO TEST FOR MULTIPLE COMMODITIES ONE EPISODE