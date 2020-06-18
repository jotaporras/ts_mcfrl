import gym
from gym import spaces
import random
import numpy as np

class EnvironmentParameters:
    def __init__(self):
        self.num_dcs

class ShippingFacilityEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ShippingFacilityEnvironment, self).__init__()
        print("Calling init on the ShippingFacilityEnvironment")
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)


    def step(self, action):
        # Execute one time step within the environment
        print(f"Stepping with action {action}")
        obs = random.randint(0, 10)
        reward = random.randint(0, 100)
        done = np.random.choice([True, False])
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        print("Reseting environment")
        return random.randint(0, 10)  # first observation.

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("rendering")

    def _next_observation(self):
        print("Calling random next obs")
        return random.randint(0, 10)


if __name__ == '__main__':
    env = ShippingFacilityEnvironment()
    first_obs = env.reset()
    for i in range(15):
        # the agent observes the first state and chooses an action
        #### todo

        #environment steps with the agent's action and returns new state and reward
        obs, reward, done, info = env.step(random.randint(0, 3))

        #Render the current state of the environment
        env.render()
