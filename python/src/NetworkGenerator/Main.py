from network.Network import Network
from orders.Orders import Orders
from NetworkGenerator.ExtendedNetwork import ExtendedNetwork
from NetworkGenerator.GraphExtension import GraphExtension

def main():
    #for node in nodes:
    #    print(node.id)
    
    net = Network(3,2,2)
    GraphExtension.GraphNetwork(net.dcs+net.costumers, net.arcs)
    
    orders = Orders(10, 3, net)
    print("Total delivery time: " + str(orders.totalTime))
    for order in orders.orders:
        print("Capacity: " + str(order.capacity) + " InitialP: " + order.initialPoint.id + " FinalP: "+ order.finalPoint.id+" DeliveryT: " + str(order.deliveryTime))


    extendedNetwork = ExtendedNetwork(net, orders)
    nodes, arcs = extendedNetwork.ConvertToExtended()
    GraphExtension.GraphNetwork(nodes, arcs)
    
if __name__ == '__main__':
    main()