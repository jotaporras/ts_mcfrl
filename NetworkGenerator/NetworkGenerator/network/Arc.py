class Arc:
    """
    Definition of an arc

    :param string id: arc identifier, must be unique
    :param Node nodeFrom: node from where the arc starts
    :param Node nodeTo: node from where the arc ends
    """
    def __init__(self, id, nodeFrom, nodeTo):
        self.id = id
        self.nodeFrom = nodeFrom
        self.nodeTo = nodeTo
