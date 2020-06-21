from typing import List, Any

from locations.Order import Order
from network.Node import Node, TENNode
from network.Arc import Arc
from network.PhysicalNetwork import PhysicalNetwork
import numpy as np


class ExtendedNetwork:
    """
    Definition of a network
    
    :param Network network:
    :param Orders fixed_orders:
    
    """
    fixed_orders: List[Order]
    open_orders: List[Order]
    network: PhysicalNetwork
    inventory: np.array

    def __init__(self, network, inventory, fixed_orders, open_orders=[]):
        self.network = network
        self.inventory = inventory
        self.fixed_orders = fixed_orders
        self.open_orders = open_orders

    def ConvertToExtended(self):
        return self.__GenerateNodes(), self.__GenerateArcs()

    def __GenerateNodes(self):  # Todo add commodities here.
        # Marks which node will have an initial order
        count = 0
        nodes_with_demand = {}
        self.location_time_nodemap = {}  # this map assumes one order per customer timestep. Node dictionary of (dc,t) -> [Node] * num_commodities
        for order in self.fixed_orders:
            nodes_with_demand[str(order.customer.node_id) + "_" + str(count)] = order.demand
            count += order.due_timestep + 1

        # Creates all the dcs nodes
        nodes = []
        node_id_inc = 0
        total_time = self.network.planning_horizon

        num_commodities = self.network.num_commodities
        for k in range(num_commodities):
            for dc in self.network.dcs:
                for t in range(total_time):
                    expanded_node_id = node_id_inc
                    dc_k_inventory = self.inventory[dc.node_id, k] if t == 0 else 0  # inventory is (dc,k)
                    node = TENNode(expanded_node_id, dc_k_inventory, 0, k, dc.kind, t, name=f"{expanded_node_id}__{dc.name}^{k}:{t}")
                    # Adds the load according to the initial point of the order
                    # if node.node_id in nodes_with_demand: # TODO what is this??????
                    #     node.flow = nodes_with_demand[node.node_id]
                    nodes.append(node)

                    self.location_time_nodemap.setdefault((dc.node_id, t), [None] * num_commodities)[k] = node

                    node_id_inc += 1

        # Creates all the customer nodes
        for order in self.fixed_orders:
            for k in range(num_commodities):
                k_demand = order.demand[k]  # type hints go crazy here
                if k_demand > 0:  # only generate nodes for commodities that have it.
                    KIND = "C"  # TODO validate if order TEN node should be of kind C (i think yes)
                    node = TENNode(node_id_inc, -k_demand, 0, k, KIND, order.due_timestep,
                                   name=f"{node_id_inc}__{order.name}^{k}:{order.due_timestep}")
                    nodes.append(node)
                    location_time_key = (order.customer.node_id, order.due_timestep)
                    self.location_time_nodemap.setdefault(location_time_key, [None] * num_commodities)[k] = node
                    node_id_inc += 1

        # todo add open orders too! (for the MCF agent)
        return nodes

    def __GenerateArcs(self):
        arcs = []
        arcsToCostumers = []

        planning_horizon = self.network.planning_horizon
        num_commodities = self.network.num_commodities
        network = self.network
        # Generate the arcs to the same dcs across t
        arc_id_incr = 0
        for dc_node in self.network.dcs:
            for k in range(num_commodities):
                for t in range(0, planning_horizon - 1):
                    tail = self.location_time_nodemap[(dc_node.node_id, t)][k]
                    head = self.location_time_nodemap[(dc_node.node_id, t + 1)][k]

                    cost = self.network.default_storage_cost
                    capacity = self.network.default_inf_capacity

                    ten_arc = Arc(arc_id_incr, tail, head, cost, capacity, k, name=f"{tail.name}=>{head.name}")
                    arcs.append(ten_arc)
                    arc_id_incr += 1
        #TODo aqui quede ver por que no estan matcheando el delivery date de la orden con el plan horizon de la corrida.
        # Generates the arcs beetwen differentes DCs over time. DCs can be connected on the same t (could be hardwired to more duration.
        transport_duration = 0  # TODO hardwired
        for physical_arc in self.network.arcs:
            for k in range(num_commodities):
                for t in range(0, planning_horizon):
                    physical_tail = physical_arc.tail
                    physical_head = physical_arc.head

                    if (physical_head.node_id, t) not in self.location_time_nodemap.keys():
                        continue

                    tail = self.location_time_nodemap[(physical_tail.node_id, t)][k]
                    head = self.location_time_nodemap[(physical_head.node_id, t)][k]

                    transport_cost = network.default_dc_transport_cost
                    capacity = self.network.default_inf_capacity  # todo i think the physical network was meant to have a capacity?
                    arc_name = f"{tail.name}=>{head.name}"

                    ten_arc = Arc(arc_id_incr, tail, head, transport_cost, capacity, k, name=arc_name)

                    # nodeFrom = Node(physical_arc.tail.arc_id + "_" + str(t), physical_arc.tail.demand, physical_arc.tail.flow)
                    # nodeTo = Node(physical_arc.head.arc_id + "_" + str(t + physical_arc.cost), physical_arc.head.demand, physical_arc.head.flow)
                    # newArc = Arc(nodeFrom.id +"_to_" + nodeTo.id, nodeFrom, nodeTo, physical_arc.cost, physical_arc.demand)
                    if physical_head.kind == "C":
                        if (physical_head.node_id, t) in self.location_time_nodemap.keys(): #if there is a
                            print("Adding an order arc")
                            arcs.append(ten_arc)
                            #arcsToCostumers.append(ten_arc)
                    else:
                        arcs.append(
                            ten_arc)  # TODO I THINK IT ALWAYS SHOULD BE ADDED TO ARCS BUT LET'S SEE WHY THIS LOGIC PREVAILED.

                        # TODO DELETE THIS WHEN PREVIOUS TODO OF ARCS AND ARCSTOCUSTOMERS IS RESOLVED.
                    # if physical_arc.cost + t < self.fixed_orders.totalTime and physical_arc.head.arc_id[0] != 'c': #TODO i think i dont need this pruning.
                    #     arcs.append(newArc)
                    # elif physical_arc.head.arc_id[0] == 'c':
                    #     arcsToCostumers.append(newArc)
                    arc_id_incr += 1

        # Filters the arcs going to costumers based in the locations delivery time #TODO see if necessary.
        count = 0
        for order in self.fixed_orders:
            for arc in arcsToCostumers:
                time = int(arc.head.arc_id[-1])
                if time == count + order.due_timestep:
                    arcs.append(arc)
                    #arcsToCostumers.remove(arc)
            count += order.due_timestep

        return arcs
