from typing import List, Any

from locations.Order import Order
from network.Node import Node, TENNode
from network.Arc import Arc
from network.PhysicalNetwork import PhysicalNetwork
import numpy as np

DEBUG=True

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

    def ConvertToExtended(self,current_t,planning_horizon_t):
        return self.__GenerateNodes(current_t, planning_horizon_t), self.__GenerateArcs(current_t, planning_horizon_t)

    def __GenerateNodes(self,current_t,planning_horizon_t):
        # Marks which node will have an initial order
        count = 0
        nodes_with_demand = {} #todo this map is never used.
        self.location_time_nodemap = {}  # this map assumes one order per customer timestep. Node dictionary of (dc,t) -> [Node] * num_commodities
        for order in self.fixed_orders:
            if current_t <= order.due_timestep <= planning_horizon_t:
                nodes_with_demand[str(order.customer.node_id) + "_" + str(count)] = order.demand
                count += order.due_timestep + 1 #TODO maybe delete this???

        # Creates all the dcs nodes
        nodes = []
        node_id_inc = 0
        total_time = self.network.planning_horizon

        num_commodities = self.network.num_commodities
        for k in range(num_commodities):
            for dc in self.network.dcs:
                # for t in range(total_time):
                for t in range(current_t, planning_horizon_t+1):
                    expanded_node_id = node_id_inc
                    dc_k_inventory = self.inventory[dc.node_id, k] if t == current_t else 0  # inventory is (dc,k)
                    node = TENNode(expanded_node_id, dc_k_inventory, 0, k, dc.kind, t, dc, name=f"{expanded_node_id}__{dc.name}^{k}:{t}")
                    # Adds the load according to the initial point of the order
                    # if node.node_id in nodes_with_demand: # TODO what is this??????
                    #     node.flow = nodes_with_demand[node.node_id]
                    nodes.append(node)

                    #print("Adding to location time nodemap for dc: ", (dc.node_id, t))
                    self.location_time_nodemap.setdefault((dc.node_id, t), [None] * num_commodities)[k] = node

                    node_id_inc += 1

        # Creates all the customer nodes
        for order in self.fixed_orders:
            if current_t <= order.due_timestep <= planning_horizon_t:
                for k in range(num_commodities):
                    k_demand = order.demand[k]  # type hints go crazy here
                    if k_demand > 0:  # only generate nodes for commodities that have it.
                        KIND = "C"
                        node = TENNode(node_id_inc, -k_demand, 0, k, KIND, order.due_timestep, order.customer,
                                       name=f"{node_id_inc}__{order.name}^{k}:{order.due_timestep}")
                        nodes.append(node)
                        location_time_key = (order.customer.node_id, order.due_timestep)
                        #print("Adding to location time nodemap for customer: ",location_time_key," on order ",order.name)
                        self.location_time_nodemap.setdefault(location_time_key, [None] * num_commodities)[k] = node
                        node_id_inc += 1

        # TODO theres something wrong with this calculation because the balancer doesnt complain. probably has to do with time windows.
        # if DEBUG:
        #     balances = np.sum(self.inventory,axis=0)-sum(map(lambda o: o.demand, self.fixed_orders))
        #     #print("balances",balances)
        #     if np.sum(balances)!=0:
        #         print("SOME BALANCES NOT 0", np.argwhere(balances != 0))
        #         print(balances)

        # todo add open orders too! (for the MCF agent)
        return nodes

    def __GenerateArcs(self,current_t,planning_horizon_t):
        arcs = []
        arcsToCostumers = []

        planning_horizon = self.network.planning_horizon
        num_commodities = self.network.num_commodities
        network = self.network
        # Generate the arcs to the same dcs across t
        arc_id_incr = 0
        for dc_node in self.network.dcs:
            for k in range(num_commodities):
                # for t in range(0, planning_horizon - 1):
                for t in range(current_t, planning_horizon_t):
                    tail = self.location_time_nodemap[(dc_node.node_id, t)][k]
                    head = self.location_time_nodemap[(dc_node.node_id, t + 1)][k]

                    cost = self.network.default_storage_cost
                    capacity = self.network.default_inf_capacity

                    ten_arc = Arc(arc_id_incr, tail, head, cost, capacity, k, name=f"{tail.name}=>{head.name}")
                    arcs.append(ten_arc)
                    arc_id_incr += 1
        # Generates the arcs beetwen differentes DCs over time. DCs can be connected on the same t (could be hardwired to more duration.
        transport_duration = 0  # TODO hardwired and not implemented
        # TODO this is wrong because it's connecting based on physical network, will add all possible paths to orders when they are fixed.
        # It should only add all possible arcs between DC-Customer if they are open orders. Consider skipping orders altogether in this loop and do on next/
        # Otherwise,. remove next loop and add filters for only valid arcs for orders.
        for physical_arc in self.network.arcs:
            for k in range(num_commodities):
                for t in range(current_t, planning_horizon_t+1):
                    physical_tail = physical_arc.tail
                    physical_head = physical_arc.head

                    if (physical_head.node_id, t) not in self.location_time_nodemap.keys() or (physical_tail.node_id, t) not in self.location_time_nodemap.keys() or physical_head.kind == "C":
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
                            #print("Adding an order arc",ten_arc)
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
            for k in range(num_commodities):
                if current_t <= order.due_timestep <= planning_horizon_t and order.demand[k]>0:
                    head = self.location_time_nodemap[(order.customer.node_id, order.due_timestep)][k]
                    tail = self.location_time_nodemap[(order.shipping_point.node_id, order.due_timestep)][k]#todo maybe modify to timestep-1
                    valid_connection = network.is_valid_arc(order.shipping_point.node_id,order.customer.node_id)
                    cost = network.default_customer_transport_cost if valid_connection else network.big_m_cost
                    # if not valid_connection:
                    #     print("Order ",order," is scheduled from an invalid shipping point, with a cost",cost)
                    # else:
                    #     print("Ship pt is valid")
                    arc = Arc(arc_id_incr,tail,head, cost, network.default_inf_capacity, k,name=f"{tail.name}=>{head.name}")
                    arcs.append(arc)
                    arc_id_incr += 1

        return arcs

#TODO check if big M costs are being assigned to invalid arcs!!!!