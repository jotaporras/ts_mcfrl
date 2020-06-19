import random
from network.Node import Node
from network.Arc import Arc

print("Network imported")
class Network:
    """
    Definition of a network
    
    :param int num_dcs: num of nodes that are not costumers
    :param int num_customers: num of nodes that are costumers
    :param int dcs_per_customer: dcs per costumer
    """
    def __init__(self, num_dcs, num_customers, dcs_per_customer, num_commodities=1):
        assert num_dcs >= dcs_per_customer

        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.dcs_per_customer = dcs_per_customer
        self.num_commodities = num_commodities
        self.dcs = []
        self.costumers = []
        self.arcs = []

        self.__Generate()

    def __Generate(self):
        """
        Generates the dcs and costumer nodes and the arcs
        """
        #Generates the dcs nodes
        for i in range(0, self.num_dcs):
            self.dcs.append(Node("dcs_" + str(i), 0, 0))

        #Generates the costumer nodes
        for i in range(0, self.num_customers):
            self.costumers.append(Node("c_" + str(i), 0, 0))
        
        #Generates the arcs between dcs and dcs
        for node1 in self.dcs:
            for node2 in self.dcs:
                if(node1.id != node2.id):
                    self.arcs.append(Arc(node1.id + "_to_" + node2.id, node1, node2, 1, 0))

        #Generate a random arc between dcs and costumers
        for costumer in self.costumers:
            counter = 0
            nodeDc = rd.choice(self.dcs, size=self.dcs_per_customer, replace=False)
            while counter < self.dcs_per_customer:
                self.arcs.append(Arc(nodeDc[counter].id + "_to_" + costumer.id, nodeDc[counter], costumer, 1, 0))
                counter += 1


