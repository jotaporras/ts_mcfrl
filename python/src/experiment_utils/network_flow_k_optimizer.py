


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
    extended_network = ExtendedNetwork(network, state['open'], state['fixed']) #todo add both and see if it's being correctly generated.

    # Generate ortools.
    total_cost = 0.0
    for k in range(network.num_commodities): #TODO one indexed commodities?
        total_cost += optimize_commodity(state, extended_network, k)
    return total_cost

def optimize_commodity(state, extended_network, k, inf_capacity=9000000):
    mcf = pywrapgraph.SimpleMinCostFlow()

    for n in extended_network.network.dcs:
        if n.commodity == k: #TODO see if replace with K independent lists of arcs.
            mcf.SetNodeSupply(n.id, n.capacity)  # todo refactor

    for n in extended_network.network.costumers:
        if n.commodity == k:
            mcf.SetNodeSupply(n.id,n.capacity) #todo refactor and check if demands are properly ste

    for a in extended_network.network.arcs: #todo is this right?
        if a.commodity == k:
            mcf.AddArcWithCapacityAndUnitCost(a.tail,a.head,inf_capacity,a.cost) #todo validate.

    print("Running optimization")
    start = time.process_time_ns()
    mcf.Solve()
    end = time.process_time_ns()
    elapsed_ms = (end - start) / 1000000
    print(f"elapsed {elapsed_ms}ms")
    print(f"elapsed {round_to_1(elapsed_ms / 1000)}s")

    if mcf.Solve() == mcf.OPTIMAL:
        print('Minimum cost:', mcf.OptimalCost())
    else:
        raise Exception("Something happened")
    return mcf.OptimalCost()
