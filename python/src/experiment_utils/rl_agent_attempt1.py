# import gym
# import tf_agents
# from gym import wrappers
# import tensorflow as tf
# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.environments import tf_py_environment,gym_wrapper
# from tf_agents.networks import q_network
# from tf_agents.utils.common import element_wise_squared_loss
#
# if __name__ == '__main__':
#         print("Creating environments")
#         train_py_env = wrappers.TimeLimit(gym.make("shipping_allocation:shipping-v0"), max_episode_steps=100)
#         print("Created train env")
#         eval_py_env = wrappers.TimeLimit(gym.make("shipping_allocation:shipping-v0"), max_episode_steps=100)
#         print("Creating eval env")
#         print("Wrapping")#TODO aqui quede TF no sirveeeeeeeeeeeeee
#
#         train_env = gym_wrapper.GymWrapper(train_py_env)#tf_py_environment.TFPyEnvironment(train_py_env)
#         eval_env = gym_wrapper.GymWrapper(eval_py_env)#tf_py_environment.TFPyEnvironment(eval_py_env)
#
#         fc_layer_params = (100,)
#         print("Calling q network")
#         q_net = q_network.QNetwork(
#                 train_env.observation_spec(),
#                 train_env.action_spec(),
#                 fc_layer_params=fc_layer_params)
#         print("creating optimizer")
#         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
#
#         print("init train step counter")
#         train_step_counter = tf.compat.v2.Variable(0)
#
#         tf_agent = dqn_agent.DqnAgent(
#                 train_env.time_step_spec(),
#                 train_env.action_spec(),
#                 q_network=q_net,
#                 optimizer=optimizer,
#                 td_errors_loss_fn = element_wise_squared_loss,
#                 train_step_counter = train_step_counter)
#
#         print("init tf agent")
#         tf_agent.initialize()