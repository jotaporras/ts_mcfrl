
from network.Network import Network
from orders.Orders import Orders
import sys

def main():
    net = Network(4,2,2)
    
    orders = Orders(10, 4, net)
    
    for order in orders.orders:
        print(order.capacity)

 


if __name__ == '__main__':
    main()