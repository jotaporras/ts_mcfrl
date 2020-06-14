import networkx as nx
import matplotlib.pyplot as plt

class GraphExtension:
    def GraphNetwork(nodes, arcs):
        G=nx.Graph()
        for node in nodes:
            G.add_node(node.id)
        
        for arc in arcs:
            G.add_edge(arc.nodeFrom.id, arc.nodeTo.id)
        
        #G.add_nodes_from(["b","c"])
        #G.add_edge(1,2)
        #edge = ("d", "e")
        #G.add_edge(*edge)
        #edge = ("a", "b")
        #G.add_edge(*edge)
        #G.add_edges_from([("a","c"),("c","d"),("d","c"), ("a",1), (1,"d"), ("a",2)])
        

        print("Nodes of graph: ")
        print(G.nodes())
        print("Edges of graph: ")
        print(G.edges())

        nx.draw(G, with_labels=True, font_weight='bold')
        plt.savefig("simple_path.png") # save as png
        plt.show() # display
