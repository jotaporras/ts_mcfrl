import random as rnd
from orders.Order import Order

class Orders:
    """
    Splits the totalCapacity into a random number of orders

    :param int totalCapacity: 
    :param int num_orders: 
    :param Network network: 
    """
    def __init__(self, totalCapacity, num_orders, network):
        assert totalCapacity >= num_orders
        self.totalTime = 0
        self.totalCapacity = totalCapacity
        self.num_orders = num_orders
        self.network = network
        self.orders = []

        self.__Generate()

    def __Generate(self):
        """
        Creates the orders by dividing the totalCapacity into num_orders
        """
        left = self.totalCapacity
        orderLeft = self.num_orders
        while orderLeft > 0:
            randCapacity = rnd.randint(1, left - orderLeft + 1)
            # Temporal delivery time generation, need to be change later
            orderTime = 3
            self.totalTime += orderTime            
            self.orders.append(Order(randCapacity, rnd.choice(self.network.dcs), rnd.choice(self.network.costumers), orderTime))
            left -= randCapacity
            orderLeft -= 1

        self.orders[0].capacity += left


        
