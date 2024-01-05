# !pip install -r requirements.txt
import math

# Importing the libraries
import numpy as np
import cv2
import networkx as nx
# import matplotlib.pyplot as plt
import os
import time
import subprocess

import pandas as pd


# To save the n_u and n_v values to a text file
def save_list_to_text_file(file_name, data_list):
    try:
        with open(file_name, 'a') as file:
            if isinstance(data_list, np.int64):
                file.write(str(data_list) + '\n')
            else:
                file.write(str(data_list) + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")


# To generate the neighbourhood values of u and v the soul of the code
def neighbourhood_values_generator(G):
    minimum_distances = dict(nx.shortest_path_length(G))  # To find the minimum distance between the nodes
    df = pd.DataFrame(minimum_distances)
    weiner_ind = np.sum(df.values) // 2

    cardinality_of_v_neighbourhood = []
    cardinality_of_u_neighbourhood = []

    for edges_of_graph in G.edges():  # It iterates through all the edges of the graph
        ux = df[edges_of_graph[0]].values
        uv = df[edges_of_graph[1]].values
        cardinality_of_u_neighbourhood.append(np.sum(ux < uv))
        cardinality_of_v_neighbourhood.append(np.sum(uv < ux))

    return cardinality_of_u_neighbourhood, cardinality_of_v_neighbourhood, weiner_ind


# It creates a list of neighbourhood values for all the images in the one class of dataset
def creating_dataset_for_different_classes(img_dir, upath, vpath, wpath):
    completion_val = 0  # To check if 2 images are done
    threshold = 0.5
    img_shape_perc = 15
    percent = img_shape_perc / 100

    for file in os.listdir(img_dir)[:]:

        if file.startswith('.'):
            continue
        brain_img = cv2.imread(os.path.join(img_dir, file), 0)
        shape1 = tuple(int(values * percent) for values in brain_img.shape)
        brain_img_1 = cv2.resize(brain_img, shape1)  # Resizing the image to 25x15 for checkng the code
        # brain_img_1 = np.int16(brain_img)     # If program runs corectly then use this line
        brain_img_1 = np.int16(brain_img_1)
        num_of_nodes = math.prod(brain_img_1.shape)
        brain_img_1_column = brain_img_1.reshape(num_of_nodes)
        # bright_mat = [0] * num_of_nodes
        # for i, val in enumerate(brain_img_1_column):
        #     bright_mat[i] = np.absolute(np.subtract(brain_img_1_column, val))

        # bright_mat = np.vstack(bright_mat)

        bright_mat = np.subtract.outer(brain_img_1_column, brain_img_1_column)

        min_val_least = np.amin(bright_mat)
        max_val_atmost = np.amax(bright_mat)

        neigh_mat = 1 - (np.subtract(bright_mat, min_val_least) / (max_val_atmost - min_val_least))
        neigh_mat = np.where(neigh_mat >= threshold, 1, 0) - np.identity(num_of_nodes)  # To remove the self loops

        # Create a graph object
        G = nx.Graph()
        # Add nodes to the graph
        num_nodes = neigh_mat.shape[0]
        G.add_nodes_from(range(num_nodes))
        # Add edges based on the adjacency matrix
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if neigh_mat[i, j] == 1:
                    G.add_edge(i, j)

        n_u, n_v, weiner_ = neighbourhood_values_generator(G)

        completion_val += 1  # If program runs correctly then remove this line
        # if completion_val == 2:  # If program runs correctly then remove this condition
        #     print("2 images done")
        #     break
        save_list_to_text_file(upath, n_u)
        save_list_to_text_file(vpath, n_v)
        save_list_to_text_file(wpath, weiner_)




# Main function to run the code
def main_function_program():
    # Load the images
    img_dir = "Alzheimers_Dataset/train/ModerateDemented"
    target_dir = "Alzheimers_graph_valued_Dataset 2/train/ModerateDemented"
    # for files_inside_tn_or_ts in os.listdir(img_dir):  # It iterates through the four classes
    #     if files_inside_tn_or_ts[0] == '.':
    #         continue
    #     #print(files_inside_tn_or_ts)
    #     target_final = os.path.join(target_dir, files_inside_tn_or_ts)
    os.makedirs(target_dir, exist_ok=True)
    # class_path = os.path.join(img_dir, files_inside_tn_or_ts)
    u_val_path = os.path.join(target_dir, 'neigh_u_val.txt')
    v_val_path = os.path.join(target_dir, 'neigh_v_val.txt')
    weiner_path = os.path.join(target_dir, 'weiner_index.txt')

    creating_dataset_for_different_classes(img_dir, u_val_path, v_val_path, weiner_path)

    print(f"File saved: '{target_dir}'")
    print("Time taken: ", time.time())



if __name__ == '__main__':
    start_time = time.time()
    print("Start Time: ", start_time)
    main_function_program()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken to run the program is: ", elapsed_time)
