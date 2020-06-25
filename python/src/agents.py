import random

# Environment and agent
import gym
import numpy as np
import tensorflow.compat.v1 as tf
from shipping_allocation.envs.network_flow_env import ActualOrderGenerator, NaiveInventoryGenerator, EnvironmentParameters, \
    ShippingFacilityEnvironment

from network.PhysicalNetwork import PhysicalNetwork

tf.disable_v2_behavior()

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


class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.9, learning_rate=0.015):
        super().__init__(env)
        self.state_size = env.observation_space.shape
        self.action_space_size = env.action_space.n
        print("State size:", self.state_size)

        self.eps = 0.05
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        tf.reset_default_graph()
        self.state_in = tf.placeholder(tf.float32, shape=self.state_size,name='state_in')
        self.action_in = tf.placeholder(tf.int32, shape=[1],name='action_in')
        self.target_in = tf.placeholder(tf.float32, shape=[1],name='target_in')

        #self.state = tf.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.transpose(tf.one_hot(self.action_in, depth=1))

        # Ya resuelve el tamaño de las capaz intermedias
        self.q_state = tf.layers.dense(self.state_in, units=self.action_space_size, name="q_table")
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action),
                                      axis=1)  # Verificar si realmente es la suma

        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
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
        state, action, next_state, reward, done = experience

        #TODO transform state into vector.
        state_vector = self.convert_state_to_vector(state)
        next_state_vector = self.convert_state_to_vector(next_state)

        q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state_vector})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next)

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
        inventory = state['inventory']
        stacked_inventory = inventory.reshape(-1,1)
        latest_open_order = state['open'][0]
        reshaped_demand = latest_open_order.demand.reshape(-1,1)
        state_vector = np.concatenate([stacked_inventory, reshaped_demand]) #TODO add to this the customer order id.
        return state_vector.transpose() #np.array((1,num_dcs*num_commodities + num_commodities))


    def __del__(self):
        self.sess.close()


if __name__ == "__main__":
    num_dcs = 5
    num_customers = 2
    num_commodities = 3
    orders_per_day = 1
    dcs_per_customer = 2
    demand_mean = 100
    demand_var = 20

    num_episodes = 100
    num_steps = 30

    physical_network = PhysicalNetwork(num_dcs, num_customers, dcs_per_customer,demand_mean,demand_var,num_commodities)
    # order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = NaiveInventoryGenerator()
    environment_parameters = EnvironmentParameters(
        physical_network, num_episodes, order_generator, generator
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = QNAgent(env)

    state = env.reset()
    reward = 0
    done = False
    print("=========== starting episode loop ===========")
    print("Initial environment: ")
    env.render()
    while not done:
        action = agent.get_action(state)
        print(f"Agent is taking action: {action}")
        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        next_state, reward, done, info = env.step(action)
        print(f"Got reward {reward} done {done}")

        agent.train((state,action,next_state,reward,done))

        state = next_state
        # Render the current state of the environment
        env.render()

        if done:
            print("===========Environment says we are DONE ===========")