


# {
#             "physical_network": self.environment_parameters.network,
#             "inventory": self.inventory,
#             "open": self.open_orders,
#             "fixed": self.fixed_orders,
#             "current_t": self.current_t,
# }
import time

from NetworkGenerator.ExtendedNetwork import ExtendedNetwork
from ortools.graph import pywrapgraph

from experiment_utils.general_utils import round_to_1


def optimize(state):
    network = state['physical_network']
    inventory = state['inventory']
    # Treat all orders as fixed.
    extended_network = ExtendedNetwork(network, inventory, fixed_orders=state['open'] + state['fixed'], open_orders=[])
    extended_nodes,arcs = extended_network.ConvertToExtended()

    # Generate ortools.
    total_cost = 0.0
    for k in range(network.num_commodities): #TODO one indexed commodities?
        total_cost += optimize_commodity(state, extended_network, k,extended_nodes,arcs)
    return total_cost

def optimize_commodity(state, extended_network, k, extended_nodes,arcs,inf_capacity=9000000):
    mcf = pywrapgraph.SimpleMinCostFlow()

    print("adding arcs and nodes")
    for n in extended_nodes:
        if n.commodity==k:
            print(f"mcf.SetNodeSupply({n.node_id},int({n.balance}))")
            mcf.SetNodeSupply(n.node_id,int(n.balance))
    for a in arcs:
        if a.commodity == k:
            print(f"mcf.AddArcWithCapacityAndUnitCost({a.tail.node_id}, {a.head.node_id}, {inf_capacity}, {a.cost})")
            mcf.AddArcWithCapacityAndUnitCost(a.tail.node_id, a.head.node_id, inf_capacity, a.cost)

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

    print("Running optimization")
    start = time.process_time_ns()
    status = mcf.Solve()
    end = time.process_time_ns()
    elapsed_ms = (end - start) / 1000000
    print(f"elapsed {elapsed_ms}ms")
    print(f"elapsed {round_to_1(elapsed_ms / 1000)}s")

    if status == mcf.OPTIMAL:
        print('Minimum cost:', mcf.OptimalCost())
    else:
        print(f"Status",status) #todo aqui quede looks unbalanced probs because customer are msissing saludios.
        raise Exception("Something happened")
    return mcf.OptimalCost()


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