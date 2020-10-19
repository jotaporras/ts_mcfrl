from envs.network_flow_env import ActualOrderGenerator, DirichletInventoryGenerator, EnvironmentParameters, \
    BiasedOrderGenerator

from network.PhysicalNetwork import PhysicalNetwork



def build_network_flow_env_parameters(environment_config, episode_length,order_gen:str):
    physical_network = PhysicalNetwork(
        num_dcs = environment_config['num_dcs'],
        num_customers = environment_config['num_customers'],
        dcs_per_customer = environment_config['dcs_per_customer'],
        demand_mean = environment_config['demand_mean'],
        demand_var = environment_config['demand_var'],
        big_m_factor = environment_config['big_m_factor'],
        num_commodities = environment_config['num_commodities'],
    )

    if order_gen=='original': # The original is independent means for each product customer.
        order_generator = ActualOrderGenerator(
            physical_network,
            environment_config['orders_per_day']

        )
    elif order_gen == 'biased': # biased is more skewed and there's correlations in products.
        order_generator = BiasedOrderGenerator(  # todo make this parameterized with a factory or something.
            physical_network,
            environment_config['orders_per_day']
        )
    else:
        raise NotImplementedError("alternatives are original and biased")

    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(physical_network, order_generator, generator, episode_length)

    return environment_parameters
