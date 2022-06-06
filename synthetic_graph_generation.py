import networkx as nx
import random
import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import time

start_time = time.time()
# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate test case graph and run CTD on both.')

# Optional argument
parser.add_argument('--G_num_nodes', type=int, default=100,
                    help='Number of nodes in the graph')
parser.add_argument('--G_density', type=float, default=0.1,
                    help='Density of edges in graph')
parser.add_argument('--weight_ratio', type=int, default=10,
                    help='Path to the pretrained model file.')
parser.add_argument('--node_number_ratio', type=float, default=0.01,
                    help='Percentage of samples used for training.')
args, unknown = parser.parse_known_args()


def generate_random_connected_graph(num_nodes, background_density, background_weight, return_flexible_edges = False):
    tree_base = nx.random_tree(n = num_nodes) # , seed=np.random.RandomState()
    G = tree_base
    
    num_edges_to_add = int(num_nodes*(num_nodes - 1)*0.5*background_density - (num_nodes - 1))
    #pick num_edges_to_add random edges to add to G
    flexible_edge_list = random.sample(list(nx.non_edges(G)), num_edges_to_add)
    G.add_edges_from(flexible_edge_list)
    nx.set_edge_attributes(G, values = background_weight, name = 'weight')
    
    assert(nx.is_connected(G)) 
    if (return_flexible_edges): 
        return (G, flexible_edge_list)
    else:
        return G
    
def find_paths(G, source_node, length, excludeSet = None):
    if excludeSet == None:
        excludeSet = set([source_node])
    else:
        excludeSet.add(source_node)
    if length==0:
        return [[source_node]]
    paths = [[source_node] + path for neighbor in G.neighbors(source_node) if neighbor not in excludeSet for path in find_paths(G,neighbor,length-1,excludeSet)]
    excludeSet.remove(source_node)
    return paths

#utility functions

def choose_S_path(G, num_S):
    all_candidates = []
    for node in G:
        all_candidates.extend(find_paths(G, node, num_S-1))
    result = random.choice(all_candidates)
    return result

def path_compare(path1, path2):
    assert( len(path1) == len(path2) )
    for c1, c2 in zip(path1, path2):
        if (c1 != c2):
            return False
    return True

def construct_test_case(G_num_nodes, G_density, G_background_weight, S_num_nodes, S_weight):
    G1 = generate_random_connected_graph(G_num_nodes, G_density, G_background_weight)
    G2, flexible_edges = generate_random_connected_graph(G_num_nodes, G_density, G_background_weight, return_flexible_edges = True)
    
    S = choose_S_path(G1, S_num_nodes)
    S_path_graph = nx.path_graph(S)
    
    filtered_edges = []
    for (a,b) in flexible_edges:
        if (a,b) in S_path_graph.edges():
            pass
        else:
            filtered_edges.append((a,b))
    flexible_edges= filtered_edges
    
    weight_dict = {e:S_weight for e in S_path_graph.edges()}
    for u, v in S_path_graph.edges():
        G1[u][v]['weight'] = S_weight 
        if ((u,v) not in G2.edges()):
            G2.add_edge(u, v)
            (a,b) = random.choice(flexible_edges)
            G2.remove_edge(a,b)
            flexible_edges.remove((a,b))
            G2[u][v]['weight'] = S_weight
    
    return (G1, G2, S)


def write_test_case_to_CTD2_input_files(G1, G2, S, out_name_G1=None, out_name_G2=None, S_path=None):

    S_string = '\n'.join([str(item) for item in S])
    S_path = 'S.csv' if not S_path else S_path
    with open(S_path, 'w') as f:
        f.write("S module\n"+S_string)

    adj_G1 = nx.to_pandas_adjacency(G1)
    adj_G2 = nx.to_pandas_adjacency(G2)

    if out_name_G1:
        adj_G1.to_csv(path_or_buf=out_name_G1,index=False)
    else:
        adj_G1.to_csv(path_or_buf='adj_G1.csv',index=False)
        
    if out_name_G2:
        adj_G2.to_csv(path_or_buf=out_name_G2,index=False)
    else:
        adj_G2.to_csv(path_or_buf='adj_G2.csv',index=False)


(G1, G2, S) = construct_test_case(G_num_nodes=args.G_num_nodes, G_density=args.G_density, G_background_weight=0.1, S_num_nodes=np.floor(args.node_number_ratio*args.G_num_nodes), S_weight=0.1*args.weight_ratio)
params = '_'.join(map(str, [args.G_num_nodes, args.G_density, args.weight_ratio, args.node_number_ratio]))
out_name_G1 = 'G1_' + params + '.csv'
out_name_G2 = 'G2_' + params + '.csv'
S_fname = 'S_' + params + '.csv'
write_test_case_to_CTD2_input_files(G1, G1, S, out_name_G1, out_name_G2, S_fname)

exec_time = (time.time() - start_time)
print("--- %s seconds ---" % exec_time)