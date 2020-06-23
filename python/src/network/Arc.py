from network.Node import Node


class Arc:
    """
    Definition of an arc

    :param string id: arc identifier, must be unique
    :param Node tail: node from where the arc starts
    :param Node head: node from where the arc ends
    :param int cost: cost of transit this arc
    :param int capacity: the load the arc can carry
    """
    arc_id: int
    tail: Node
    head: Node
    cost: int
    capacity: int

    def __init__(self, arc_id: int, tail, head, cost, capacity, commodity, name=""):
        self.arc_id = arc_id
        self.tail = tail
        self.head = head
        self.cost = cost
        self.capacity = capacity
        self.commodity = commodity
        self.name = name

    def transportation_arc(self)->bool:
        return self.tail.kind=="DC"and self.head.kind=="DC" and  self.tail.location.node_id != self.head.location.node_id

    def __repr__(self):
        # return f"Arc(arc_id={self.arc_id}, tail={self.tail}, head={self.head}, cost={self.cost}, capacity={self.capacity}, commodity={self.commodity}, name={self.name})"
        return self.name #short repr.