# !pip install -r requirements.txt

# Importing the libraries
import numpy as np
import cv2
import networkx as nx 
# import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import math



# To create a directory(for train n test n four classes) if it does not exist
def check_if_directory_exists(folder_path):
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)


# To save the n_u and n_v values to a text file
def save_list_to_text_file(file_name, data_list):
    try:
        with open(file_name, "a") as file:
                file.write(str(data_list) + '\n')
                # file.write("hi" + '\n')
        # print(f"List saved to '{file_name}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# To generate the neighbourhood values of u and v the soul of the code
def neighbourhood_values_generator(G):
    minimum_distances = dict(nx.shortest_path_length(G))     # To find the minimum distance between the nodes
    minimum_distances = pd.DataFrame(list(minimum_distances.values()))  # To convert the dictionary to array
    cardinality_of_v_neighbourhood = []
    cardinality_of_u_neighbourhood = []

    for edges_of_graph in G.edges():             # It iterates through all the edges of the graph
        u_valuer = minimum_distances[edges_of_graph[0]]    # To store the distance of all the nodes from u
        v_valuer = minimum_distances[edges_of_graph[1]]    # To store the distance of all the nodes from v
        n_u_value = np.sum(u_valuer < v_valuer)            # To find the number of nodes nearer to u than v
        n_v_value = np.sum(u_valuer > v_valuer)            # To find the number of nodes nearer to v than u
        cardinality_of_v_neighbourhood.append(n_v_value)   # To store the number of nodes nearer to v than u
        cardinality_of_u_neighbourhood.append(n_u_value)   # To store the number of nodes nearer to u than v
    #     the_val_u, the_val_v = [], []            # To store the set of nodes nearer to u and v respectively

        # for vertex_x_in_G in range(G.number_of_nodes()):      # It iterates through all the nodes of the graph
        #     n_u_value = 
    #         if minimum_distances[edges_of_graph[0]][vertex_x_in_G] < minimum_distances[edges_of_graph[1]][vertex_x_in_G]:
    #             the_val_u.append(vertex_x_in_G)      # Adds to val_u if the distance of x to u is less than x to v
    #         elif minimum_distances[edges_of_graph[0]][vertex_x_in_G] > minimum_distances[edges_of_graph[1]][vertex_x_in_G]:
    #             the_val_v.append(vertex_x_in_G)     # Adds to val_v if the distance of x to v is less than x to u
    #     cardinality_of_v_neighbourhood.append(len(the_val_v)) 
    #     cardinality_of_u_neighbourhood.append(len(the_val_u))   
    return cardinality_of_u_neighbourhood, cardinality_of_v_neighbourhood


# It creates a list of neighbourhood values for all the images in the one class of dataset
def creating_dataset_for_different_classes(img_dir):
    completion_val = 0         # To check if 2 images are done
    the_stored_value = 0
    tr = 0.5                   # Threshold value
    img_shape_perc = 15
    for file in os.listdir(img_dir):
        if file.startswith('.') and the_stored_value < 2514:
            the_stored_value += 1
            continue
        brain_img = cv2.imread(img_dir + '/' + file, cv2.IMREAD_GRAYSCALE)
        shape1 = []
        for values in brain_img.shape:
            shape1.append(int(values * img_shape_perc / 100))
        shape1 = tuple(shape1)
        brain_img_1 = cv2.resize(brain_img, shape1)       # Resizing the image to 25x15 for checkng the code
        brain_img_1 = np.int16(brain_img_1)
        num_of_nodes = math.prod(shape1)
        brain_img_1_column = brain_img_1.reshape(num_of_nodes)
        

        # bright_mat = np.vstack(bright_mat)
        bright_mat = np.subtract.outer(brain_img_1_column, brain_img_1_column)
        min_val_least = np.amin(bright_mat)
        max_val_atmost = np.amax(bright_mat)

        neigh_mat = 1 - (np.subtract(bright_mat, min_val_least)/ (max_val_atmost - min_val_least))
        # neigh_mat[neigh_mat >= 0.5] = 1
        # neigh_mat[neigh_mat != 1]  = 0
        neigh_mat1 = np.where(neigh_mat < tr, 0, 1)
        neigh_mat = neigh_mat1 - np.identity(num_of_nodes)   # To remove the self loops 
        G = nx.Graph()
        for i in range(num_of_nodes):
            for j in range(i+1, num_of_nodes):
                if neigh_mat[i][j] == 1:
                    G.add_edge(i, j)
        n_u_all, n_v_all = neighbourhood_values_generator(G)
        # completion_val += 1     # If program runs corectly then remove this line
        # if completion_val == 200:    # If program runs corectly then remove this condition  
        #     print("images done")
        #     break
        save_list_to_text_file("Alzheimers_Dataset_points/train/NonDemented/neigh_u_val.txt", n_u_all)   # To save the neighbourhood values as text file
        save_list_to_text_file("Alzheimers_Dataset_points/train/NonDemented/neigh_v_val.txt", n_v_all)   # To save the neighbourhood values as text file
    #     n_u_all.append(n_u)
    #     n_u_all.append(n_v)
    print("Done")
    # return n_u_all, n_v_all, weiner_index


# Main function to run the code
def main_function_program():
    # Load the images
    img_dir = "Alzheimers_Dataset"
    target_dir = "Alzheimers_graph_valued_Dataset"
    for files in os.listdir(img_dir):       # It iterates through the train and test folders
        if files.startswith('.'):
            continue
        img_dir_tn_or_tst = img_dir + '/' + files
        target_dir_tn_or_tst = target_dir + '/' + files 
        for files_inside_tn_or_ts in os.listdir(img_dir_tn_or_tst):  # It iterates through the four classes
            if files_inside_tn_or_ts.startswith('.'):
                continue
            print(files_inside_tn_or_ts)
            target_final = target_dir_tn_or_tst + '/' + files_inside_tn_or_ts
            check_if_directory_exists(target_final)
            neigh_u_val, neigh_v_val, weiner_index = creating_dataset_for_different_classes(img_dir_tn_or_tst + '/' + files_inside_tn_or_ts)
            save_list_to_text_file(target_final + '/' + 'neigh_u_val.txt', neigh_u_val)   # To save the neighbourhood values as text file
            save_list_to_text_file(target_final + '/' + 'neigh_v_val.txt', neigh_v_val)   # To save the neighbourhood values as text file
            save_list_to_text_file(target_final + '/' + 'weiner_index.txt', weiner_index) # To save the weiner index as text file  

if __name__ == '__main__':
    start_time = time.time()
    # main_function_program()
    creating_dataset_for_different_classes("Alzheimers_Dataset/train/NonDemented")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken to run the program is: ", elapsed_time)
    

