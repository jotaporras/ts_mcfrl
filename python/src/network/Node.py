class Node:
    """
    Definition of a node
    
    :param string name: node identifier, must be unique
    :param int balance: store capacity
    ::param int load: the current load carried
    """
    def __init__(self, id, balance, flow, commodity, kind,name=""):
        self.node_id = id
        self.balance = balance
        self.flow = flow
        self.commodity = commodity
        self.name = name
        self.kind = kind

    def __repr__(self):
        return self.name


class TENNode(Node):
    def __init__(self, id, balance, flow, commodity,kind,time,name):
        super().__init__(id, balance, flow, commodity,kind,name)
        self.time = time