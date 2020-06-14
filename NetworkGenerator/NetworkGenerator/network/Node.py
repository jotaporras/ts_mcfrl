class Node:
    """
    Definition of a node
    
    :param string name: node identifier, must be unique
    :param int capacity: store capacity
    ::param int load: the current load carried
    """
    def __init__(self, id, capacity, load):
        self.id = id
        self.capacity = capacity
        self.load = load

