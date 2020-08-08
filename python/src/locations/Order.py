import numpy as np

from network.Node import Node


class Order:
    """
    Definition of an order

    :param int demand: demand vector of size K.
    :param Node shipping_point: the initial node from where it will start
    :param Node customer: the final node where it should be delivered
    :param int delivery_time: the time it takes from shipping point to customer.
    """
    shipping_point: Node
    customer: Node
    demand: np.array# (k,1)
    due_timestep: int

    def __init__(self, demand: np.array, shipping_point, customer, delivery_time, name):
        self.demand = demand
        self.shipping_point = shipping_point
        self.customer = customer
        self.due_timestep = delivery_time
        self.name = name

    def __repr__(self):
        return f"Order(demand={self.demand}, shipping_point={self.shipping_point}, customer={self.customer}, deliveryTime={self.due_timestep})"