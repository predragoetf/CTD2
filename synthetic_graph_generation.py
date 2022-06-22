import networkx as nx
import random
import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import time
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

start_time = time.time()
# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate test case graph and run CTD on both.')

# Optional argument
parser.add_argument('--G_num_nodes', type=int, default=100,
                    help='Number of nodes in the graph')
parser.add_argument('--G_density', type=float, default=0.1,
                    help='Density of edges in graph')
parser.add_argument('--weight_ratio', type=int, default=33,
                    help='Path to the pretrained model file.')
parser.add_argument('--node_number_ratio', type=float, default=0.1,
                    help='Percentage of samples used for training.')
parser.add_argument('--S_graph_type', type=int, default=0,
                    help='Type of the planted subgraph. Clique (0), path graph (1).')
args, unknown = parser.parse_known_args()


def generate_random_connected_graph(num_nodes, background_density, background_weight, return_flexible_edges = False):
    eprint("Start tree generation")
    start_time = time.time()
    tree_base = nx.random_tree(n = num_nodes) # , seed=np.random.RandomState()
    exec_time = (time.time() - start_time)
    eprint(f"Tree generation lasted {exec_time} seconds")
    G = tree_base
    
    num_edges_to_add = int(num_nodes*(num_nodes - 1)*0.5*background_density - (num_nodes - 1))
    #pick num_edges_to_add random edges to add to G
    flexible_edge_list = random.sample(list(nx.non_edges(G)), num_edges_to_add)
    #G = nx.Graph(set(G.edges).union(set(flexible_edge_list)))
    G.add_edges_from(flexible_edge_list)
    nx.set_edge_attributes(G, values = background_weight, name = 'weight')
    
    assert(nx.is_connected(G)) 
    if (return_flexible_edges): 
        ret = (G, flexible_edge_list)
    else:
        ret = G
    eprint("Finished graph generation")
    return ret
    
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


def simple_depth_walk(G, source_node, length, stack, exclude_set=set()):
    if length == 0:
        ret = stack
    else:
        exclude_set.add(source_node)
        candidates = set(G.neighbors(source_node)).difference(exclude_set)
        if len(candidates) > 0:
            next_node = random.choice()
            new_stack = stack.push(source_node)
            ret = simple_depth_walk(G, next_node, length - 1, new_stack, exclude_set)
        else:
            ret = None
    return ret


def rewire(G, S_graph, S_weight):

    #add missing edges from S_path
    #weight_dict = {e: S_weight for e in S_graph.edges()}
    added_edges = []
    for u, v in S_graph.edges():
        if (u, v) not in G.edges():
            added_edges.append((u, v))
            G.add_edge(u, v)
        G[u][v]['weight'] = S_weight

    eprint("Added all S edges to graph")
    #construct a spanning tree to be able to remove edges
    source_node = random.choice(list(G.nodes))
    spanning_dfs_tree = nx.dfs_tree(G, source=source_node, depth_limit=len(G))

    eprint("Constructed a spannig tree")
    protected_edges = set(S_graph.edges()).union(set(spanning_dfs_tree.edges()))
    unprotected_edges = set(G.edges()).difference(protected_edges)

    edges_to_erase = random.sample(list(unprotected_edges), len(added_edges))

    eprint("Started edge removal")
    #new_G = nx.Graph(set(G.edges).difference(edges_to_erase))
    G.remove_edges_from(edges_to_erase)
    eprint("Removed edges and finished graph rewiring")
    #return new_G
    return G

def choose_S_path(G, num_S):
    #candidates = []
    S_path = None
    spins = 0
    visited_start_nodes = []
    path = None
    #while ((len(candidates)==0) and (spins < len(G))):
    while (not S_path and (spins < len(G))):
        eprint(f"spin:{spins}")
        a = G.nodes
        start_node = random.choice(list(G.nodes()))
        if start_node in visited_start_nodes:
            continue
        else:
            visited_start_nodes.append(start_node)

        #candidates = find_paths(G, start_node, num_S-1)

        S_path = None
        walks = 0
        while not S_path and walks<len(G):
            S_path = simple_depth_walk(G, start_node, num_S - 1, [])
            walks += 1

        # if len(candidates)>0:
        #     path = random.choice(candidates)
        #     break
        if S_path:
            path = S_path
        else:
            spins += 1
    return path

    # Legacy code for path generation, picking from all existing paths
    # all_candidates = []
    # for node in G:
    #     all_candidates.extend(find_paths(G, node, num_S-1))
    # result = random.choice(all_candidates)
    # return result


def path_compare(path1, path2):
    assert( len(path1) == len(path2) )
    for c1, c2 in zip(path1, path2):
        if (c1 != c2):
            return False
    return True


def construct_test_case(G_num_nodes, G_density, G_background_weight, S_num_nodes, S_weight, S_graph_type):
    graph_type = {0: 'clique', 1: 'path'}
    G1 = generate_random_connected_graph(G_num_nodes, G_density, G_background_weight)
    G2, flexible_edges = generate_random_connected_graph(G_num_nodes, G_density, G_background_weight, return_flexible_edges = True)
    
    S_nodes = random.sample(list(G1.nodes), S_num_nodes)
    if graph_type[S_graph_type] == 'path':
        S_graph = nx.path_graph(S_nodes)
    else:
        S_graph = nx.complete_graph(S_nodes)
        
    rewire(G1, S_graph, S_weight)
    
    rewire(G2, S_graph, S_weight)

    assert nx.is_connected(G1), "G1 is not connected"
    assert nx.is_connected(G2), "G2 is not connected"
    
    return (G1, G2, list(S_graph.nodes) )


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


(G1, G2, S) = construct_test_case(G_num_nodes=args.G_num_nodes, G_density=args.G_density, G_background_weight=0.1, S_num_nodes=int(np.floor(args.node_number_ratio*args.G_num_nodes)), S_weight=0.1*args.weight_ratio, S_graph_type=args.S_graph_type)
params = '_'.join(map(str, [args.G_num_nodes, args.G_density, args.weight_ratio, args.node_number_ratio]))
out_name_G1 = 'G1_' + params + '.csv'
out_name_G2 = 'G2_' + params + '.csv'
S_fname = 'S_' + params + '.csv'
write_test_case_to_CTD2_input_files(G1, G1, S, out_name_G1, out_name_G2, S_fname)

exec_time = (time.time() - start_time)
eprint("Written graphs: {} | {}".format(out_name_G1, out_name_G2))
eprint("--- %s seconds ---" % exec_time)

# tree = nx.random_tree(n=10, seed=0)
# eprint(nx.forest_str(tree, sources=[0]))
# S_test = nx.path_graph([0,3,4])
# eprint(S_test)
# tree = rewire(tree, S_test, 5)
#eprint(nx.forest_str(tree, sources=[0]))

# import matplotlib.pyplot as plt
#
# nx.draw(tree, with_labels = True)
# plt.show()

#TODO:
# 1. Add all missing edges from S to G (let there be k of those)
# 2. run DFS from any node and obtain a spanning tree
# 3. erase k edges from 