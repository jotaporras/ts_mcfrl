


# {
#             "physical_network": self.environment_parameters.network,
#             "inventory": self.inventory,
#             "open": self.open_orders,
#             "fixed": self.fixed_orders,
#             "current_t": self.current_t,
# }
import time
import numpy as np

from NetworkGenerator.ExtendedNetwork import ExtendedNetwork
from ortools.graph import pywrapgraph

from experiment_utils.general_utils import round_to_1
from network.PhysicalNetwork import PhysicalNetwork
import logging

DEBUG=False

def optimize(state):
    network:PhysicalNetwork = state['physical_network']
    inventory = state['inventory']
    current_t = state['current_t']
    planning_horizon_t = current_t + network.planning_horizon - 1
    # Treat all orders as fixed.
    #extended_network = ExtendedNetwork(network, inventory, fixed_orders=state['open'] + state['fixed'], open_orders=[])
    # only fixed
    extended_network = ExtendedNetwork(network, inventory, fixed_orders=state['fixed'], open_orders=[])
    extended_nodes, arcs = extended_network.ConvertToExtended(current_t, planning_horizon_t)

    inv_shape = inventory.shape
    transport_matrix = np.zeros(inv_shape)

    # Generate ortools.
    total_cost = 0.0
    total_big_m_count = 0 #TODO: should it be just one? Because if you have it in one commodity you have it in all, so it's kind of the same in different scale.
    for k in range(network.num_commodities): #TODO one indexed commodities?
        k_cost,tm,all_movements,big_m_counter = optimize_commodity(state, extended_network, k, extended_nodes, arcs, current_t,inv_shape)
        total_cost += k_cost
        transport_matrix += tm

    if DEBUG:
        logging.info(f"Total optimization cost: {total_cost}")
        logging.info("Total transportation movements: ")
        logging.info(transport_matrix)

    return total_cost, transport_matrix, all_movements, big_m_counter

def optimize_commodity(state, extended_network, k, extended_nodes,arcs,current_t,inventory_shape,inf_capacity=9000000):
    mcf = pywrapgraph.SimpleMinCostFlow()
    slack_node_id = len(extended_nodes)

    #logging.info("adding arcs and nodes")
    problem_balance = 0
    mcfarcs = {}
    for n in extended_nodes:
        if n.commodity==k:
            #logging.info(f"mcf.SetNodeSupply({n.node_id},int({n.balance})), node: {n.name},{n}")
            mcf.SetNodeSupply(n.node_id,int(n.balance))
            problem_balance+=n.balance
    for a in arcs:
        if a.commodity == k:
            #logging.info(f"mcf.AddArcWithCapacityAndUnitCost({a.tail.node_id}, {a.head.node_id}, {inf_capacity}, {a.cost}), arc: {a.name},{a}")
            mcfarcs[(a.tail.node_id, a.head.node_id)] = a
            mcf.AddArcWithCapacityAndUnitCost(a.tail.node_id, a.head.node_id, inf_capacity, a.cost)

    # This was the first attempt at handling imbalanced problems, but then I coded it directly into the orders.
    # TODO delete if the other way works.+
    # if problem_balance !=0:
    #     logging.info(problem_balance)
    #     # balance with a slack node, connect to all arcs.
    #     # TODO DELETE THIS OF FIX WITH THE NEW DUMMY NODES WORK.
    #     mcf.SetNodeSupply(slack_node_id,-int(problem_balance))
    #     if problem_balance > 0:
    #         #excess inventory, connect as outbound to slack. We only really need to connect the ones that have inventory
    #         for n in extended_nodes:
    #             if n.balance>0:
    #                 mcf.AddArcWithCapacityAndUnitCost(n.node_id, slack_node_id, inf_capacity, 1)#unit cost, see if correct unit
    #     else:
    #         #lacking, this should never happen.
    #         raise Exception(f"Encountered unbalanced problem ({problem_balance} units) lacking inventory on {k}. On current assumptions this should never happen.")



    if problem_balance != 0:
        logging.warning(f"WARN!! MCF balance for {k} is {problem_balance}")



    #TODO ai think delete.
    # for n in extended_network.network.dcs:
    #     if n.commodity == k: #TODO see if replace with K independent lists of arcs.
    #         mcf.SetNodeSupply(n.arc_id, n.demand)  # todo refactor
    #
    # for n in extended_network.network.customers:
    #     if n.commodity == k:
    #         mcf.SetNodeSupply(n.arc_id, n.demand) #todo refactor and check if demands are properly set
    #
    # for a in extended_network.network.arcs: #todo is this right?
    #     if a.commodity == k:
    #         mcf.AddArcWithCapacityAndUnitCost(a.tail,a.head,inf_capacity,a.cost) #todo validate.

    #logging.info("Running optimization")
    start = time.process_time()
    status = mcf.Solve()
    end = time.process_time()
    elapsed_ms = (end - start) / 1000000
    #logging.info(f"elapsed {elapsed_ms}ms")
    #logging.info(f"elapsed {round_to_1(elapsed_ms / 1000)}s")

    transport_movements = np.zeros(inventory_shape)
    all_movements = []
    big_m_counter = 0
    if status == mcf.OPTIMAL:
        #logging.info("\nFlows: ")
        for ai in range(mcf.NumArcs()):
            tail = mcf.Tail(ai)
            head = mcf.Head(ai)
            a = mcfarcs[(tail, head)]

            # Accumulate all movements occurring at current_t
            if a.commodity==k and mcf.Flow(ai) > 0 and a.head.time==current_t:
                all_movements.append((a,mcf.Flow(ai)))

            #logging.info(f"{a.name} = {mcf.Flow(ai)}",end="")
            if a.commodity==k and a.transportation_arc() and mcf.Flow(ai)>0 and a.head.time==current_t:
                transport_movements[a.tail.location.node_id,k] -= mcf.Flow(ai) #subtract from source
                transport_movements[a.head.location.node_id, k] += mcf.Flow(ai)  #add to destination
            if a.cost >= state['physical_network'].big_m_cost and mcf.Flow(ai)>0:
                big_m_counter+=1
                if DEBUG:
                    logging.info(f"This is a Big M cost found in the optimization {a} ==> {mcf.Flow(ai)}")
                    logging.info(f"{a.tail.location}, {a.head.location}")
            #if a.commodity==k and a.transportation_arc() and mcf.Flow(ai)>0:
                #logging.info("***")
                #logging.info(f"***This a transp arc id {a.arc_id} with flow",a,mcf.Flow(ai)) #toido aqui quede y ver bien flows.
            #else:
               # logging.info("")
        # logging.info('Minimum cost:', mcf.OptimalCost())

    else:
        logging.info(f"Status",status)
        raise Exception("Something happened") #todo aqui quede the small onehot is failing with the new multicommodity. It's yielding infeasible. Could be reversed inv.

    #if (transport_movements>0).any():
        #logging.info("Executing an inventory transport: ")
        #logging.info(transport_movements)

    return mcf.OptimalCost(),transport_movements, all_movements,big_m_counter


# MinCostFlowBase_BAD_COST_RANGE = 6
#
# MinCostFlowBase_BAD_RESULT = 5
#
# MinCostFlowBase_FEASIBLE = 2
# MinCostFlowBase_INFEASIBLE = 3
#
# MinCostFlowBase_NOT_SOLVED = 0
#
# MinCostFlowBase_OPTIMAL = 1
# MinCostFlowBase_UNBALANCED = 4