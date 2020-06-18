class Arc:
    """
    Definition of an arc

    :param string id: arc identifier, must be unique
    :param Node nodeFrom: node from where the arc starts
    :param Node nodeTo: node from where the arc ends
    :param int cost: cost of transit this arc
    :param int capacity: the load the arc can carry
    """
    def __init__(self, id, nodeFrom, nodeTo, cost, capacity):
        self.id = id
        self.nodeFrom = nodeFrom
        self.nodeTo = nodeTo
        self.cost = cost
        self.capacity = capacity
