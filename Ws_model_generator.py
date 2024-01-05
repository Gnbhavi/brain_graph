import numpy as np
import networkx as nx

class Ws_generator:
    def __init__(self, n, k):
       self.num_vertices = n
       self.G = nx.Graph()
       if k % 2 != 0:
           raise ValueError
       self.halfk = int(k/2)
       for intial_vertex in range(self.num_vertices):
           edge_of_i = [(intial_vertex, (intial_vertex+end_vertex+1) % self.num_vertices ) 
                        for end_vertex in range(self.halfk)]
           self.G.add_edges_from(edge_of_i)
       pass

    def rewiring_ws(self, p):
        i = 0
        while True:
            i += 1
            G1 = self.G.copy()
            for i in range(1, self.halfk+1):
                for j in range(self.num_vertices):
                    if np.random.uniform(0, 1) < p:
                        w = np.random.randint(0, self.num_vertices - 1)
                        if j != w and w not in self.G[j]:
                            G1.add_edge(j, w)
                            G1.remove_edge(j, np.mod(i+j, self.num_vertices))
            if nx.is_connected(G1) or i > 100:
                break
        return G1


def neighbourhood_values_generator(G):
    minimum_distances = dict(nx.shortest_path_length(G))
    cardinality_of_v_neighbourhood = []
    cardinality_of_u_neighbourhood = []
    for edges_of_graph in G.edges():
        the_val_u, the_val_v = [], []
        for vertices_in_G in range(G.number_of_nodes()):
            if minimum_distances[edges_of_graph[0]][vertices_in_G] < minimum_distances[edges_of_graph[1]][vertices_in_G]:
                the_val_u.append(vertices_in_G)
            elif minimum_distances[edges_of_graph[0]][vertices_in_G] > minimum_distances[edges_of_graph[1]][vertices_in_G]:
                the_val_v.append(vertices_in_G)
        cardinality_of_v_neighbourhood.append(len(the_val_v)) 
        cardinality_of_u_neighbourhood.append(len(the_val_u))   
    return cardinality_of_u_neighbourhood, cardinality_of_v_neighbourhood