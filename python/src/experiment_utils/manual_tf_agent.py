# from random import random
#
# import tensorflow as tf
# import numpy as np
# import gym
# import time
#
#
# class Agent():
#     def __init__(self, env):
#         self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
#
#         if self.is_discrete:
#             self.action_size = env.action_space.n
#             print("Action size:", self.action_size)
#         else:
#             self.action_low = env.action_space.low
#             self.action_high = env.action_space.high
#             self.action_shape = env.action_space.shape
#             print("Action range:", self.action_low, self.action_high)
#
#     def get_action(self, state):
#         if self.is_discrete:
#             action = random.choice(range(self.action_size))
#         else:
#             action = np.random.uniform(self.action_low,
#                                        self.action_high,
#                                        self.action_shape)
#         return action
#
# # Tired of TF agents, imma do it myself with an example from Openai gym
# class DQNAgent(Agent):
#     def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
#         super().__init__(env)
#         self.state_size = env.observation_space.shape[0]
#         print("State size:", self.state_size)
#
#         self.eps = 1.0
#         self.discount_rate = discount_rate
#         self.learning_rate = learning_rate
#         self.build_model()
#
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
#     def build_model(self):
#         tf.reset_default_graph()
#         self.state_in = tf.placeholder(tf.int32, shape=[1]) #todo change this shape
#         self.action_in = tf.placeholder(tf.int32, shape=[1])
#         self.target_in = tf.placeholder(tf.float32, shape=[1])
#
#         self.state = tf.one_hot(self.state_in, depth=self.state_size)
#         self.action = tf.one_hot(self.action_in, depth=self.action_size)
#
#         self.q_state = tf.layers.dense(self.state, units=self.action_size, name="q_table")
#         self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)
#
#         self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
#         self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#
#     def get_action(self, state):
#         q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
#         action_greedy = np.argmax(q_state)
#         action_random = super().get_action(state)
#         return action_random if random.random() < self.eps else action_greedy
#
#     def train(self, experience):
#         state, action, next_state, reward, done = ([exp] for exp in experience)
#
#         q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state})
#         q_next[done] = np.zeros([self.action_size])
#         q_target = reward + self.discount_rate * np.max(q_next)
#
#         feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
#         self.sess.run(self.optimizer, feed_dict=feed)
#
#         if experience[4]:
#             self.eps = self.eps * 0.99
#
#     def __del__(self):
#         self.sess.close()
#
#
# if __name__ == '__main__': # TODO AQUI QUEDE MAKE THIS WORK ALSO JUPYTER DONT RECOGNIZE TF.
#     env = gym.make("shipping_allocation:shipping-v0")
#     agent = DQNAgent(env)
#     num_episodes = 3
#
#     for ep in range(num_episodes):
#         state = env.reset()
#         total_reward = 0
#         done = False
#         while not done:
#             action = agent.get_action(state)
#             next_state, reward, done, info = env.step(action)
#             agent.train(state, action, next_state, reward, done)
#             env.render()
#             total_reward += reward
#             state = next_state
# ###
# ###
# ###
# ###
# ###
# ###
# ###
# ###
# from random import random
#
# import tensorflow as tf
# import numpy as np
# import gym
# import time
#
#
# class Agent():
#     def __init__(self, env):
#         self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
#
#         if self.is_discrete:
#             self.action_size = env.action_space.n
#             print("Action size:", self.action_size)
#         else:
#             self.action_low = env.action_space.low
#             self.action_high = env.action_space.high
#             self.action_shape = env.action_space.shape
#             print("Action range:", self.action_low, self.action_high)
#
#     def get_action(self, state):
#         if self.is_discrete:
#             action = random.choice(range(self.action_size))
#         else:
#             action = np.random.uniform(self.action_low,
#                                        self.action_high,
#                                        self.action_shape)
#         return action
#
# # Tired of TF agents, imma do it myself with an example from Openai gym
#     def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
#         super().__init__(env)
#         self.state_size = env.observation_space.shape[0]
#         print("State size:", self.state_size)
#
#         self.eps = 1.0
#         self.discount_rate = discount_rate
#         self.learning_rate = learning_rate
#         self.build_model()
#
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
# tf.reset_default_graph()
# state_size=10*5
# state_in = tf.placeholder(tf.int32, shape=[1]) #todo change this shape
# action_in = tf.placeholder(tf.int32, shape=[1])
# target_in = tf.placeholder(tf.float32, shape=[1])
#
# state = tf.one_hot(state_in, depth=state_size)
# action = tf.one_hot(action_in, depth=action_size)
#
# q_state = tf.layers.dense(state, units=action_size, name="q_table")
# q_action = tf.reduce_sum(tf.multiply(q_state, action), axis=1)
#
# loss = tf.reduce_sum(tf.square(target_in - q_action))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#
#     def get_action(self, state):
#         q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
#         action_greedy = np.argmax(q_state)
#         action_random = super().get_action(state)
#         return action_random if random.random() < self.eps else action_greedy
#
#     def train(self, experience):
#         state, action, next_state, reward, done = ([exp] for exp in experience)
#
#         q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state})
#         q_next[done] = np.zeros([self.action_size])
#         q_target = reward + self.discount_rate * np.max(q_next)
#
#         feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
#         self.sess.run(self.optimizer, feed_dict=feed)
#
#         if experience[4]:
#             self.eps = self.eps * 0.99
#
#     def __del__(self):
#         self.sess.close()
#
#
# if __name__ == '__main__': # TODO AQUI QUEDE MAKE THIS WORK ALSO JUPYTER DONT RECOGNIZE TF.
#     env = gym.make("shipping_allocation:shipping-v0")
#     agent = DQNAgent(env)
#     num_episodes = 3
#
#     for ep in range(num_episodes):
#         state = env.reset()
#         total_reward = 0
#         done = False
#         while not done:
#             action = agent.get_action(state)
#             next_state, reward, done, info = env.step(action)
#             agent.train(state, action, next_state, reward, done)
#             env.render()
#             total_reward += reward
#             state = next_state
