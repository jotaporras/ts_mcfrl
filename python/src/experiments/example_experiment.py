from experiment_utils import experiment_runner


if __name__ == "__main__":
    num_dcs = 5
    num_customers = 7
    num_commodities = 3
    orders_per_day = 3
    dcs_per_customer = 3
    demand_mean = 100
    demand_var = 20

    num_steps = 50
    num_episodes = 100
    #runner = experiment_runner.create_random_experiment_runner(num_dcs,num_customers,dcs_per_customer,demand_mean,demand_var,num_commodities,orders_per_day)
    runner = experiment_runner.create_always_first_dc_agent(num_dcs,num_customers,dcs_per_customer,demand_mean,demand_var,num_commodities,orders_per_day)
    runner.run_episodes(num_steps,num_episodes,orders_per_day,experiment_name='dumb_agent')
    # experiment_runner.run_episodes(
    #     num_dcs,
    #     num_customers,
    #     dcs_per_customer,
    #     demand_mean,
    #     demand_var,
    #     num_commodities,
    #     orders_per_day,
    #     num_steps,
    #     num_episodes,
    #     experiment_name="example_random_experiment",
    # )
