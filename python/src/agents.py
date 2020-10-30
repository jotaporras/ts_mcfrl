import logging
import random

# Environment and agent
from typing import List

import gym
import numpy as np
import tensorflow.compat.v1 as tf
from shipping_allocation.envs.network_flow_env import ActualOrderGenerator, NaiveInventoryGenerator, EnvironmentParameters, \
    ShippingFacilityEnvironment

from experiments_v2.q_learning_agent import ShippingEnvQLearningAgent
from locations.Order import Order
from locations import Orders
from network.PhysicalNetwork import PhysicalNetwork

tf.disable_v2_behavior()
DEBUG=False

def get_greedy_agent(env, agent_name:str):
    if agent_name == "random":
        return RandomAgent(env)
    elif agent_name == "always_zero":
        return AlwaysZeroAgent(env)
    elif agent_name == "best_fit":
        return BestFitAgent(env)
    elif agent_name == "random_valid":
        return RandomValid(env)
    elif agent_name == "do_nothing":
        return DoNothingAgent(env)
    elif agent_name == "agent_highest":
        return AgentHighest(env)
    else:
        raise NotImplementedError(f"Agent {agent_name} not implemented.")

class Agent:
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete

        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)

    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low,
                                       self.action_high,
                                       self.action_shape)
        return action
    def train(self, experience):
        pass


class RandomAgent(Agent):
    """The world's simplest agent!"""

    def train(self, experience):
        pass #do nothing

class AlwaysZeroAgent(Agent):
    """The world's dumbest agent!"""

    def get_action(self, state):
        return 0

    def train(self, experience):
        pass #do nothing

class BestFitAgent(Agent):
    """The world's most conservative agent!"""
    env:ShippingFacilityEnvironment
    network:PhysicalNetwork
    def __init__(self, env):
        super().__init__(env)
        self.env=env
        self.network = env.environment_parameters.network


    def get_action(self, state):
        inventory = state['inventory']
        order = state['open'][0]
        customer = order.customer
        cid = customer.node_id-self.network.num_dcs
        cust_dcs = np.argwhere(self.network.dcs_per_customer_array[cid, :] > 0)[:,0]
        allowed_dc_invs = inventory[cust_dcs,:]
        demand = order.demand
        remaining  = np.sum(allowed_dc_invs-demand,axis=1)
        chosen_dc_index= np.argmax(remaining)
        chosen_dc_id = cust_dcs[chosen_dc_index]

        if DEBUG:
            print("Bestfit chose: ", chosen_dc_id)
            print("Inventories: ", inventory)
            print("Allowed DCs:", cust_dcs)

            if self.network.dcs_per_customer_array[cid,chosen_dc_id] == 1:
                print("Chose allowed DC:",cid,chosen_dc_index)
            else:
                print("Chose ILLEGAL OH NO DC:", cid, chosen_dc_index)
            if np.argwhere(cust_dcs==chosen_dc_id).size==0:
                print("BESTFIT CHOSE ILLEGAL MOVEMENT. THIS SHOULD NOT HAPPEN. Illegal for customer ",customer,"DC",chosen_dc_id)
            else:
                print("Bestfit chose the legal move",chosen_dc_id)


        return chosen_dc_id#todo test this.

    def train(self, experience):
        pass #do nothing


class RandomValid(Agent):
    """The world's least screwup random agent!"""
    env: ShippingFacilityEnvironment
    network: PhysicalNetwork

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.environment_parameters.network

    def get_action(self, state):
        inventory = state['inventory']
        order = state['open'][0]
        customer = order.customer
        cid = customer.node_id - self.network.num_dcs
        cust_dcs = np.argwhere(self.network.dcs_per_customer_array[cid, :] > 0)[:,0]
        chosen_dc_id = np.random.choice(cust_dcs)

        if DEBUG:
            logging.debug(f"RandomValid chose:  {chosen_dc_id}")
            logging.debug(f"Inventories:  {inventory}")
            logging.debug(f"Allowed DCs: {cust_dcs}")
            logging.debug(f"Chose allowed DC {chosen_dc_id} for customer {cid}: {self.network.dcs_per_customer_array[cid, chosen_dc_id] == 1}")

        return chosen_dc_id  # todo test this.

    def train(self, experience):
        pass  # do nothing

class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.9, learning_rate=0.015):
        super().__init__(env)
        self.state_size = env.observation_space.shape[1]
        self.action_space_size = env.action_space.n
        print("State size:", self.state_size)

        self.eps = 0.05 # bump up to see whats up
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        tf.reset_default_graph()
        self.state_in = tf.linalg.normalize(tf.placeholder(tf.float32, shape=[1,self.state_size],name='state_in'),axis=1)[0]
        self.action_in = tf.placeholder(tf.int32, shape=[1],name='action_in')
        self.target_in = tf.placeholder(tf.float32, shape=[1],name='target_in')

        #self.state = tf.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.transpose(tf.one_hot(self.action_in, depth=self.action_space_size))

        # Ya resuelve el tamaño de las capaz intermedias
        self.l1=tf.layers.dense(self.state_in,units=500,activation=tf.nn.relu,kernel_initializer=tf.initializers.glorot_normal())
        self.l2=tf.layers.dense(self.l1,units=250,activation=tf.nn.relu,kernel_initializer=tf.initializers.glorot_normal())
        self.l3=tf.layers.dense(self.l1,units=25,activation=tf.nn.relu,kernel_initializer=tf.initializers.glorot_normal())
        self.q_state = tf.layers.dense(self.l2, units=self.action_space_size, name="q_table") # (actions*5,actions)

        #self.q_state = tf.layers.dense(self.state_in, units=self.action_space_size, name="q_table")
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action),
                                      axis=1)  # Verificar si realmente es la suma

        self.loss = tf.reduce_sum(self.target_in - self.q_action)
        # ADAM algorithm
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def get_action(self, state): #prob no hay que tocarlo.
        state_vector = self.convert_state_to_vector(state)
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in: state_vector})
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        chosen_action = action_random if random.random() < self.eps else action_greedy
        return chosen_action

    def train(self, experience):
        #print("agent.train()")
        state, action, next_state, reward, done = experience

        #TODO transform state into vector.
        state_vector = self.convert_state_to_vector(state)
        next_state_vector = self.convert_state_to_vector(next_state)

        q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state_vector})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next)

        if True:
            print("DEBUG NETWORK ON STEP",state['current_t'])
            print("reward: ",reward)
            # print("Q target: ",q_target)
            # print("Q next: ",q_next)

            #q_next,loss,target_in,q_action = self.sess.run([self.q_state,self.loss,self.target_in,self.q_action], feed_dict={self.state_in: next_state_vector})
            #q_next,loss = self.sess.run([self.q_state,self.loss], feed_dict={self.state_in: next_state_vector})
            q_state_debug,loss,l1,l2,q_action,act = self.sess.run([self.q_state,self.loss,self.l1,self.l2,self.q_action, self.action], feed_dict={self.state_in: state_vector,
                                                                             self.action_in: [action],
                                                                                   self.target_in: [q_target]})
            print("Q next: ", q_state_debug)
            # print("Q q_action: ", q_action)
            # print("Q action: ",act)
            # #print("layers",l1,l2)
            print("Q ACTION",q_action)
            print("Experience action", act)
            print("Action INPUT", action)
            print("TARGET INPUT",q_target)
            print("====")
            # print("Current loss", loss)




        feed = {self.state_in: state_vector, self.action_in: [action], self.target_in: [q_target]}
        self.sess.run(self.optimizer, feed_dict=feed)

        if done:
            self.eps = self.eps * 0.99

    # {
    #     "physical_network": self.environment_parameters.network,
    #     "inventory": self.inventory.copy(),
    #     "open": self.open_orders.copy(),
    #     "fixed": self.fixed_orders.copy(),
    #     "current_t": self.current_t,
    # }
    def convert_state_to_vector(self, state):
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


    def __del__(self):
        self.sess.close()

class DoNothingAgent(Agent):
    """The world's least screwup random agent!"""
    env: ShippingFacilityEnvironment
    network: PhysicalNetwork

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.environment_parameters.network

    def get_action(self, state):
        order = state['open'][0]
        dc = order.shipping_point

        return dc.node_id

    def train(self, experience):
        pass  # do nothing

class AgentHighest(Agent):
    """The world's debugging agent"""
    env: ShippingFacilityEnvironment
    network: PhysicalNetwork

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.environment_parameters.network

    def get_action(self, state):
        order = state['open'][0]
        customer = order.customer
        cid = self.network.num_dcs - customer.node_id
        cust_dcs = np.argwhere(self.network.dcs_per_customer_array[cid, :] > 0)[:, 0]

        return cust_dcs[-1] # choose the last

    def train(self, experience):
        pass  # do nothing








# if __name__ == "__main__":
#     num_dcs = 5
#     num_customers = 2
#     num_commodities = 3
#     orders_per_day = 1
#     dcs_per_customer = 2
#     demand_mean = 100
#     demand_var = 20
#
#     num_episodes = 100
#     num_steps = 30
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
#     print("=========== starting episode loop ===========")
#     print("Initial environment: ")
#     env.render()
#     while not done:
#         action = agent.get_action(state)
#         print(f"Agent is taking action: {action}")
#         # the agent observes the first state and chooses an action
#         # environment steps with the agent's action and returns new state and reward
#         next_state, reward, done, info = env.step(action)
#         print(f"Got reward {reward} done {done}")
#
#         agent.train((state,action,next_state,reward,done))
#
#         state = next_state
#         # Render the current state of the environment
#         env.render()
#
#         if done:
#             print("===========Environment says we are DONE ===========")