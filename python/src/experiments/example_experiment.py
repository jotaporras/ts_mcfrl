from experiment_utils import experiment_runner

def small_validation_experiments():
    num_dcs = 10
    num_customers = 5
    num_commodities = 5
    orders_per_day = 3
    dcs_per_customer = 2
    demand_mean = 100
    demand_var = 20

    num_steps = 15
    num_episodes = 5

    runner_random = experiment_runner.create_random_experiment_runner(num_dcs, num_customers, dcs_per_customer,
                                                                      demand_mean, demand_var, num_commodities,
                                                                      orders_per_day, num_steps)
    runner_dqn = experiment_runner.create_dqn_experiment_runner(num_dcs, num_customers, dcs_per_customer, demand_mean,
                                                                demand_var, num_commodities, orders_per_day, num_steps)
    runner_zero = experiment_runner.create_alwayszero_experiment_runner(num_dcs, num_customers, dcs_per_customer,
                                                                        demand_mean,
                                                                        demand_var, num_commodities, orders_per_day,
                                                                        num_steps)
    runner_bestfit = experiment_runner.create_bestfit_experiment_runner(num_dcs, num_customers, dcs_per_customer,
                                                                        demand_mean,
                                                                        demand_var, num_commodities, orders_per_day,
                                                                        num_steps)
    # runner = experiment_runner.create_always_first_dc_agent(num_dcs, num_customers,dcs_per_customer,demand_mean,demand_var,num_commodities,orders_per_day,num_steps)
    # runner_random.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='dumb_agent_unittest')
    # runner_dqn.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='dqn_agent_unittest')
    # runner_zero.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='zero_agent_unittest')
    runner_bestfit.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='bestfit_agent_unittest') #todo aqui quedé testear la red grande.

def run_large_experiment():
    num_dcs = 10
    num_customers = 50
    num_commodities = 5
    orders_per_day = 10
    dcs_per_customer = 3
    demand_mean = 200
    demand_var = 20

    num_steps = 30
    num_episodes = 1

    runner_random = experiment_runner.create_random_experiment_runner(num_dcs, num_customers, dcs_per_customer,
                                                                      demand_mean, demand_var, num_commodities,
                                                                      orders_per_day, num_steps)
    runner_dqn = experiment_runner.create_dqn_experiment_runner(num_dcs, num_customers, dcs_per_customer, demand_mean,
                                                                demand_var, num_commodities, orders_per_day, num_steps)
    runner_zero = experiment_runner.create_alwayszero_experiment_runner(num_dcs, num_customers, dcs_per_customer, demand_mean,
                                                                demand_var, num_commodities, orders_per_day, num_steps)
    runner_bestfit = experiment_runner.create_bestfit_experiment_runner(num_dcs, num_customers, dcs_per_customer,
                                                                        demand_mean,
                                                                        demand_var, num_commodities, orders_per_day,
                                                                        num_steps)                                                            
    #runner_random.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='dumb_agent')
    runner_dqn.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='dqn_agent')
    #runner_zero.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='zero_agent')
    #runner_bestfit.run_episodes(num_steps, num_episodes, orders_per_day, experiment_name='bestfit_agent') #todo aqui quedé testear la red grande.

if __name__ == "__main__":
    #small_validation_experiments()
    run_large_experiment()