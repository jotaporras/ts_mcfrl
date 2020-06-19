
class Order:
    """
    Definition of an order

    :param int capacity: the size of the order
    :param Node initialPoint: the initial node from where it will start
    :param Node finalPoint: the final node where it should be delivered
    :param int deliveryTime: the available time for the order
    """
    def __init__(self, capacity, initialPoint, finalPoint, deliveryTime):
        self.capacity = capacity
        self.initialPoint = initialPoint
        self.finalPoint = finalPoint
        self.deliveryTime = deliveryTime

    def __repr__(self):
        return f"Order(capacity={self.capacity}, initialPoint={self.initialPoint}, finalPoint={self.finalPoint}, deliveryTime={self.deliveryTime})"