from network.Node import Node
from network.Arc import Arc

class ExtendedNetwork:
    """
    Definition of a network
    
    :param Network network:
    :param Orders orders: 
    
    """
    def __init__(self, network, orders):
        self.network = network
        self.orders = orders

    def ConvertToExtended(self):
        return self.__GenerateNodes(), self.__GenerateArcs()

    def __GenerateNodes(self):
        #Marks which node will have an initial order
        count = 0
        nodesWithCap = {}
        for order in self.orders.orders:
            nodesWithCap[order.initialPoint.id+"_"+str(count)] = order.capacity
            count += order.deliveryTime + 1

        print(nodesWithCap)

        #Creates all the dcs nodes
        arcs = []
        for t in range(0, self.orders.totalTime):
            for dcs in self.network.dcs:
                node = Node(dcs.id + "_" + str(t), dcs.capacity, dcs.load)
                #Adds the load according to the initial point of the order
                if node.id in nodesWithCap:
                    node.load = nodesWithCap[node.id]
                arcs.append(node)
        
        #Creates all the costumer nodes
        for t in range(1, self.orders.totalTime):
            for costumer in self.network.costumers:
                node = Node(costumer.id + "_" + str(t), costumer.capacity, costumer.load)
                arcs.append(node)

        return arcs

    def __GenerateArcs(self):
        arcs = []
        arcsToCostumers = []
        #Generate the arcs to the same dcs across t
        for t in range(0, self.orders.totalTime-1):
            for node in self.network.dcs:
                nodeFrom = Node(node.id+"_"+str(t), node.capacity, node.load)
                nodeTo = Node(node.id+"_"+str(t+1), node.capacity, node.load)
                arc = Arc(nodeFrom.id+"_"+nodeTo.id, nodeFrom, nodeTo, 1, 1)
                arcs.append(arc)

        #Generates the arcs beetwen differentes nodes
        for t in range(0, self.orders.totalTime):
            for arc in self.network.arcs:
                nodeFrom = Node(arc.nodeFrom.id+"_"+str(t), arc.nodeFrom.capacity, arc.nodeFrom.load)
                nodeTo = Node(arc.nodeTo.id+"_"+str(t+arc.cost), arc.nodeTo.capacity, arc.nodeTo.load)
                newArc = Arc(nodeFrom.id+"_to_"+nodeTo.id, nodeFrom, nodeTo, arc.cost, arc.capacity)
                if arc.cost + t < self.orders.totalTime and arc.id[0] != 'c':
                    arcs.append(newArc)
                elif arc.id[0] == 'c':
                    arcsToCostumers.append(newArc)
        
        #Filters the arcs going to costumers based in the orders delivery time
        count = 0
        for order in self.orders.orders:
            for arc in arcsToCostumers:
                time = int(arc.nodeTo[-1])
                if time == count + order.deliveryTime and time < self.orders.totalTime:
                    arcs.append(arc)
                    arcsToCostumers.remove(arc)
            count += order.deliveryTime
        return arcs
