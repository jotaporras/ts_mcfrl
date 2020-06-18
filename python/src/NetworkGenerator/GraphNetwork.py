import matplotlib.pyplot as plt

class GraphNetwork(object):
    """description of class"""
    def Plot(network):
        ax = plt.axes()
        ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.show()


