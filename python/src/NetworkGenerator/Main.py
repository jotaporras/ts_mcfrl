from network.PhysicalNetwork import PhysicalNetwork
from locations.Orders import Orders
from NetworkGenerator.ExtendedNetwork import ExtendedNetwork
from NetworkGenerator.GraphExtension import GraphExtension

def main():
    #for node in nodes:
    #    print(node.id)
    
    net = PhysicalNetwork(3, 2, 2, demand_mean=100, demand_var=20,num_commodities=3)
    GraphExtension.GraphNetwork(net.dcs + net.customers, net.arcs)
    
    orders = Orders(10, net)
    print("Total delivery time: " + str(orders.totalTime))
    for order in orders.orders:
        print("Capacity: " + str(order.demand) + " InitialP: " + order.initialPoint.arc_id + " FinalP: " + order.finalPoint.arc_id + " DeliveryT: " + str(order.due_timestep))


    # extendedNetwork = ExtendedNetwork(net, orders)
    # nodes, arcs = extendedNetwork.ConvertToExtended()
    # GraphExtension.GraphNetwork(nodes, arcs)
    
if __name__ == '__main__':
    main()