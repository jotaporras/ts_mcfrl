import random as rnd
from locations.Order import Order
import numpy as np

from network.PhysicalNetwork import PhysicalNetwork


class Orders:
    """
    Splits the totalCapacity into a random number of locations

    :param int num_orders:
    :param Network network: 
    """
    # def __init__(self, totalCapacity, num_orders, network):
    def __init__(self, num_orders, network: PhysicalNetwork):
        self.totalTime = 0
        # self.totalCapacity = totalCapacity
        self.num_orders = num_orders
        self.network = network
        self.orders = []

        self.generate_orders()

    # def __Generate(self):
    #     """
    #     Creates the locations by dividing the totalCapacity into num_orders
    #     """
    #     left = self.totalCapacity
    #     orderLeft = self.num_orders
    #     while orderLeft > 0:
    #         randCapacity = rnd.randint(1, left - orderLeft + 1)
    #         # Temporal delivery time generation, need to be change later
    #         orderTime = 3
    #         self.totalTime += orderTime
    #         self.orders.append(Order(randCapacity, rnd.choice(self.network.dcs), rnd.choice(self.network.customers), orderTime))
    #         left -= randCapacity
    #         orderLeft -= 1
    #
    #     self.orders[0].capacity += left

    def generate_orders(self):
        # Choose orders to be generated this timestep.


        demand_var = self.network.demand_var
        # Generate demand
        demand = np.random.multivariate_normal([200,100,50],np.eye(3)*demand_var,size=(10,5)) #(num_customers,commodities,orders)


        
