import networkx as nx
from synthetic_graph_generation import rewire
from synthetic_graph_generation import write_test_case_to_CTD2_input_files
import numpy as np

def construct_real_test_case(G1_path, G2_path, S_path, S_weight_mul, S_graph_type):
    G1_df = pd.read_csv(G1_path)
    G1_df.index = G1_df.columns
    G1=nx.from_pandas_adjacency(G1_df)
    G2_df = pd.read_csv(G2_path)
    G2_df.index = G2_df.columns
    G2=nx.from_pandas_adjacency(G2_df)
    
    G1_pos = G1_df.applymap(np.abs)
    average_weight = sum(G1_pos.sum()) / sum((G1_pos != 0).sum(1))
    print(sum((G1_pos != 0).sum(1)))
    print(average_weight)
    
    S_nodes = pd.read_csv(S_path)
    S_nodes = S_nodes[S_nodes.columns[0]].values

    if S_graph_type == 'path':
        S_graph = nx.path_graph(S_nodes)
    else:
        S_graph = nx.complete_graph(S_nodes)
        
    rewire(G1, S_graph, average_weight * S_weight_mul)
    rewire(G2, S_graph, average_weight * S_weight_mul)
    return (G1, G2, list(S_graph.nodes) )


G1, G2, S = construct_real_test_case('diseases/arg.csv', 'diseases/rcdp.csv', 'diseases/s_arg.csv', 2, 'clique')
write_test_case_to_CTD2_input_files(G1, G1, S, 'diseases/arg_mod_2.csv', 'diseases/rcdp_mod_2.csv', 'diseases/s_arg_mod_2.csv')

G1, G2, S = construct_real_test_case('diseases/arg.csv', 'diseases/rcdp.csv', 'diseases/s_arg.csv', 3, 'clique')
write_test_case_to_CTD2_input_files(G1, G1, S, 'diseases/arg_mod_3.csv', 'diseases/rcdp_mod_3.csv', 'diseases/s_arg_mod_3.csv')

G1, G2, S = construct_real_test_case('diseases/arg.csv', 'diseases/rcdp.csv', 'diseases/s_arg.csv', 5, 'clique')
write_test_case_to_CTD2_input_files(G1, G1, S, 'diseases/arg_mod_5.csv', 'diseases/rcdp_mod_5.csv', 'diseases/s_arg_mod_5.csv')