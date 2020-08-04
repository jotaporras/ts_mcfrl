import random
from typing import List

from locations.Order import Order
from network.Node import Node
from network.Arc import Arc
import numpy as np
import random as rd

print("Network imported")

#For debugging, mostly
class Customer:
    def __init__(self, customer_id, allowed_dc_indices):
        self.customer_id = customer_id
        self.customer_name = f"C_{customer_id}"
        self.allowed_dc_indices = allowed_dc_indices
        self.kind="Customer"

    def __repr__(self):
        return f"Customer(customer_id={self.customer_id}, customer_name={self.customer_name}, allowed_dc_indices={self.allowed_dc_indices})"

class DistributionCenter:
    def __init__(self, dc_id):
        self.dc_id = dc_id
        self.dc_name = f"W_{dc_id}"
        self.kind = "DC"

    def __repr__(self):
        return f"DistributionCenter(dc_id={self.dc_id}, dc_name={self.dc_name})"

class PhysicalNetwork:
    """
    Definition of a network
    
    :param int num_dcs: num of nodes that are not costumers
    :param int num_customers: num of nodes that are costumers
    :param int dcs_per_customer: dcs per costumer
    """
    dcs: List[Node]
    customers: List[Node]
    dcs_debug: List[DistributionCenter]
    customers_debug: List[Customer]
    inventory_dirichlet_parameters: np.array
    planning_horizon: int

    def __init__(self, num_dcs, num_customers, dcs_per_customer, demand_mean,demand_var, num_commodities=1,planning_horizon=5):
        print("Calling physical network gen")
        self.default_storage_cost = 1 #TODO HARDWIRED CONSTANTS
        self.default_delivery_time = 3 #TODO HARDWIRED CONSTANTS
        #self.default_dc_transport_cost = 10 #TODO HARDWIRED CONSTANTS
        self.default_dc_transport_cost = 15 #TODO HARDWIRED CONSTANTS
        self.default_customer_transport_cost = 15 #TODO HARDWIRED CONSTANTS
        self.default_inf_capacity = 999999
        self.big_m_cost = self.default_customer_transport_cost*100000
        self.demand_var = demand_var
        self.demand_mean = demand_mean
        self.planning_horizon = planning_horizon
        assert num_dcs >= dcs_per_customer

        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.dcs_per_customer = dcs_per_customer
        self.num_commodities = num_commodities
        self.dcs = []
        self.dcs_debug = []
        self.customers = []
        self.customers_debug = []
        self.arcs = []
        self.dcs_per_customer_array: np.array = None
        self.inventory_dirichlet_parameters = None

        self._generate()

    def _generate(self):
        """
        Generates the dcs and costumer nodes and the arcs
        """

        # Generate allowed DCs per customer
        base_dc_assignment = np.zeros(self.num_dcs)
        base_dc_assignment[0:self.dcs_per_customer] = 1

        self.dcs_per_customer_array = np.array([np.random.permutation(base_dc_assignment) for c in range(self.num_customers)]) #Shape (num_customers,num_dcs)

        #Generates the dcs nodes
        location_id=0
        for i in range(self.num_dcs):
            self.dcs.append(Node(location_id, 0, 0,-1,kind="DC",name="dcs_" + str(location_id)))
            self.dcs_debug.append(DistributionCenter(i))
            location_id+=1

        #Generates the customer nodes
        for i in range(self.num_customers):
            self.customers.append(Node(location_id, 0, 0, 1,kind="C",name="c_" + str(location_id)))
            self.customers_debug.append(Customer(i,self.dcs_per_customer_array[i,:]))
            location_id+=1

        #Generates the arcs between dcs and dcs
        arc_id = 0
        for node1 in self.dcs:
            for node2 in self.dcs:
                if node1.node_id != node2.node_id:
                    self.arcs.append(Arc(arc_id, node1, node2, 1, 0, 1,name=str(node1.node_id) + "_to_" + str(node2.node_id)))
                    arc_id+=1

        #heavymetal distribution.
        total_demand_mean = self.demand_mean * self.num_customers * self.num_commodities
        # self.demand_mean_matrix = np.floor(
        #     np.random.dirichlet(self.num_customers / np.arange(1, self.num_customers + 1),
        #                         size=1) * total_demand_mean).reshape(-1) #(cust)
        self.demand_mean_matrix = np.floor(
            np.random.dirichlet([5.]*self.num_customers,
                                size=1) * total_demand_mean).reshape(-1)  # (cust)

        # Generate distribution parameters for the customers.
        # self.customer_means = np.random.poisson(self.demand_mean, size=self.num_customers)
        self.customer_means = np.floor(
            np.random.dirichlet(self.num_customers / np.arange(1, self.num_customers + 1),
                                size=1) * total_demand_mean).reshape(-1)+self.demand_mean#(cust) #sum mean at the end to avoid negs.

        print("Current customer means",self.customer_means)

        # Parameters for inventory distribution hardwired for now.
        self.inventory_dirichlet_parameters = [5.] * self.num_dcs #todo deprecated not used

        # TODO probably delete this.
        #Generate a random arc between dcs and costumers
        for cid in range(self.num_customers):
            #counter = 0
            #nodeDc = np.random.choice(self.dcs, size=self.dcs_per_customer, replace=False)
            customer_node = self.customers[cid]
            for dc in np.argwhere(self.dcs_per_customer_array[cid,:]>0).reshape(-1):
                dcO_node = self.dcs[dc]
                cost = self.default_customer_transport_cost
                capacity = self.default_inf_capacity
                self.arcs.append(Arc(arc_id, dcO_node, customer_node, cost, capacity, -1, name=f"{dcO_node.name}_to_{customer_node.name}"))
                arc_id+=1
                #counter += 1

    def generate_orders(self, orders_per_day: int, current_t): #TODO test and validate.
        # Choose customers to generate orders with OUT replacement, orders per day must be <= customers
        chosen_customers = np.random.choice(np.arange(self.num_customers), size=orders_per_day, replace=False)
        order_means = self.customer_means[chosen_customers]

        demand = np.floor(np.random.multivariate_normal(order_means, np.eye(orders_per_day) * self.demand_var,
                                      size=self.num_commodities))  # shape (num_commodities,num_orders)
        if (demand<0).any():
            print("Customer means that caused negatives")
            print(order_means)
            #raise Exception("Generated a negative order")
            demand = np.abs(demand)
        # Create order objects
        orders = []
        for ci in range(len(chosen_customers)):
            order_demand_vector = demand[:,ci]
            _chosen_customer = chosen_customers[ci]
            customer_node = self.customers[_chosen_customer]
            chosen_initial_point = np.random.choice(np.argwhere(self.dcs_per_customer_array[ci, :]).reshape(-1)) #TO DO AQUI QUEWDE HACER ESTO.
            initial_point_physical_node = self.dcs[chosen_initial_point]
            time = current_t+self.planning_horizon-1 #Orders appear on the edge of PH.
            orders.append(Order(order_demand_vector, initial_point_physical_node, customer_node, time, name=f"oc_{customer_node.node_id}:{time}"))
        #print("Orders",orders)
        return orders

    def is_valid_arc(self,dc,customer):
        base_customer_id=customer-self.num_dcs
        return self.dcs_per_customer_array[base_customer_id,dc]==1

    def __repr__(self):
        return str(self.__dict__)