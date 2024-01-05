import numpy as np
from numpy.random import seed
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

G_1 = nx.dense_gnm_random_graph(1000, 15000, seed=10)
G = nx.dense_gnm_random_graph(10, 21, seed=10)
D = nx.dominating_set(G, None)
print(D)
nx.draw_circular(G, with_labels=True)
plt.show()
