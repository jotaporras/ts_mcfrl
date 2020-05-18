class Node:
    """
    Definition of a node
    
    :param string name: node identifier, must be unique
    :param int capacity: store capacity
    """
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity

