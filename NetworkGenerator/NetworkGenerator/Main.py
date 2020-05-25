
from network.Network import Network
from orders.Orders import Orders
from ExtendedNetwork import ExtendedNetwork
import networkx as nx

def main():
    net = Network(2,2,2)
    
    orders = Orders(10, 4, net)
    print("Total delivery time: " + str(orders.totalTime))
    for order in orders.orders:
        print("Capacity: " + str(order.capacity) + " InitialP: " + order.initialPoint.id + " DeliveryT: " + str(order.deliveryTime))

    extendedNetwork = ExtendedNetwork(net, orders)
    nodes, arcs = extendedNetwork.ConvertToExtended()
    for node in nodes:
        print(node.id)
 


if __name__ == '__main__':
    main()