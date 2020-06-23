class Node:
    """
    Definition of a node
    
    :param string name: node identifier, must be unique
    :param int balance: store capacity
    ::param int load: the current load carried
    """
    def __init__(self, id, balance, flow, commodity, kind,location=None,name=""):
        self.node_id = id
        self.balance = balance
        self.flow = flow
        self.commodity = commodity
        self.name = name
        self.kind = kind #DC or C
        self.location = location #None if this is a physical node.

    def __repr__(self):
        return self.name+"  b="+str(int(self.balance))


class TENNode(Node):
    def __init__(self, id, balance, flow, commodity,kind,time,location,name):
        super().__init__(id, balance, flow, commodity,kind,location,name)
        self.time = time