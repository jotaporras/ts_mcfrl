from envs.network_flow_env import EnvironmentParameters, NaiveOrderGenerator, NaiveInventoryGenerator
from gym.envs.registration import register

from network.PhysicalNetwork import PhysicalNetwork

register(
    id='shipping-v0',
    entry_point='shipping_allocation.envs:ShippingFacilityEnvironment',
    kwargs={
        'environment_parameters': EnvironmentParameters(PhysicalNetwork(5, 5, 2,100, 20), 5,
                                                        NaiveOrderGenerator(5, 5, 2),
                                                        NaiveInventoryGenerator())
    }
)