import networkx as nx
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import seaborn as sns
sns.set_style("white")

G = nx.Graph()

G.add_node('SC2', pos=(5464.4, 7994.1))
G.add_node('Out1', pos=(5709.092, 5581.592))
G.add_node('S5', pos=(5728.775, 7778.673))
G.add_node('S7', pos=(5721.243, 6064.08))

G.add_edge('S7', 'Out1')
G.add_edge('S5', 'S7')
G.add_edge('SC2', 'S5')
volume = {'S5': 1000.0, 'S7': 1000.0, 'SC2': 100.0, 'Out1': 100.0}
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G,
                       pos,
                       ['S5'],
                       node_color="#88c3fa",
                       node_size=[volume['S5']],
                       label='Pond 1')
nx.draw_networkx_nodes(G,
                       pos,
                       ['S7'],
                       node_color='#43cd80',
                       node_size=[volume['S7']],
                       label='Pond 2')
nx.draw_networkx_nodes(G,
                       pos,
                       ['SC2'],
                       node_color='#D43629',
                       node_shape='+',
                       node_size=200.0,
                       label='Network Inlet')
nx.draw_networkx_nodes(G,
                       pos,
                       ['Out1'],
                       node_color='#C4A155',
                       node_size=200.0,
                       label='Network Outlet')
nx.draw_networkx_edges(G,
                       pos, edge_color='#4682B4',
                       width=2.0)
plt.axis('off')
plt.legend(numpoints = 1)
plt.show()
