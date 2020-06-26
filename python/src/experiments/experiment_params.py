
#import this file to run the final experiments

#PARAMETER MATRIX FROM EXCEL
dcs_list = [10, 25, 50, 75, 75]
num_commodities_list = [3, 5, 10, 15, 20]
customers_list = [200, 1000, 1500, 2500, 10000]
orders_per_day_list = [50, 500, 800, 1000, 2500]
dcs_per_customer_list = [2, 5, 10, 15, 15]

#OTHER PARAMS AND UTILS.
num_steps = 50
num_episodes = 200
num_experiments = len(dcs_list)