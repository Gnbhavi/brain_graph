import Ws_model_generator as wsg
import networkx as nx
import pandas as pd
import numpy as np

def values_generator_depending_on_n(num_vertex):
    ws = wsg.Ws_generator(num_vertex, 6)
    minimum_distances = dict(nx.shortest_path_length(ws.G))
    graph = ws.G
    the_neighbourhood_vertices = {}
    cardinality_of_v_neighbourhood = []
    for k_values in range(ws.halfk):
        the_neighbourhood_vertices['n_u_0' + str(k_values+1)] = []
        the_neighbourhood_vertices['n_v_0' + str(k_values+1)] = []
        for final_vertex_of_edge in range(ws.num_vertices):
            if minimum_distances[0][final_vertex_of_edge] < minimum_distances[k_values+1][final_vertex_of_edge]:
                the_neighbourhood_vertices['n_u_0' + str(k_values+1)].append(final_vertex_of_edge)
            elif minimum_distances[0][final_vertex_of_edge] > minimum_distances[k_values+1][final_vertex_of_edge]:
                the_neighbourhood_vertices['n_v_0' + str(k_values+1)].append(final_vertex_of_edge) 
        # cardinality_of_v_neighbourhood['n_u_0' + str(k_values+1)] = len(the_neighbourhood_vertices['n_u_0' + str(k_values+1)])  
        print("The set of vertices in n_u_0" + str(k_values+1) + " is ", the_neighbourhood_vertices['n_u_0' + str(k_values+1)])
        cardinality_of_v_neighbourhood.append(len(the_neighbourhood_vertices['n_u_0' + str(k_values+1)]))
    return graph, cardinality_of_v_neighbourhood
    # return graph, the_neighbourhood_vertices

def myformula_for_finding_n_u_1(n, k):
    A = int(np.floor((n - 1)/k))
    # A1 = int(np.floor((n - 2)/2))
    n_v = [0] * (int(k/2))
    h = int(k/2)
    mod_rem = np.mod(n + 1, k)
    # mod_rem = np.mod(n, h)
    rem_val = [the_rem for the_rem in range(0, int(k/2))]
    # print(mod_rem)
    for i in range(int(k/2)):
        # print(rem_val[:i+1])
        if mod_rem not in rem_val[:i+1]:
            # print("if")
            n_v[i] = (i+1) * A + 1
        else:
            # print("else")
            n_v[i] = (i+1) * A + 1 - (i + 1 - mod_rem)
            # n_v[i] = (i + 1) * A + 1 - (i + 1 - mod_rem + 1)
    return n_v

def validating_my_formula_and_orginal():
    the_table_of_n_u_v = {}
    K = {}
    valus21 = 0
    print("running")
    n = 17
    # for n in range(10, 500):
    # for n in range(7, 13):
    K[n], the_table_of_n_u_v[n] = values_generator_depending_on_n(n)
    print("hi")
    if myformula_for_finding_n_u_1(n,6) != the_table_of_n_u_v[n]:
        print("n is ", n)
        print(the_table_of_n_u_v[n])
        print(myformula_for_finding_n_u_1(n,6))
        valus21 += 1
    print(valus21)


validating_my_formula_and_orginal()