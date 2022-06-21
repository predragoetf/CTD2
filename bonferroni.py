import argparse
import json
import time
import os
import numpy as np
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

start_time = time.time()
# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate test case graph and run CTD on both.')

# Optional argument
parser.add_argument('--json_G1', type=str,
                    help='Number of nodes in the graph')
parser.add_argument('--json_G2', type=str,
                    help='Density of edges in graph')
args, unknown = parser.parse_known_args()

def calculate_bonferroni_pvalue(json_G1, json_G2, out_result_path='bonferroni.json'):
    # Loading with json.load directly from file can cause problems on Windows, so load the contents to a string first, then call loads
    # Another way to prevent this problem is to save the file as .txt instead of .json
    # Beauty of Windows...
    with open(json_G1, 'r') as json_file1, open(json_G2, 'r') as json_file2:
        contents = json_file2.read()
        data2 = json.loads(contents)
        pval2 = data2['p_value']
        bs2 = data2['optimal_bitstring']
        eprint(f"pval2 : {pval2}")

        contents = json_file1.read()
        data = json.loads(contents)
        pval1 = data['p_value']
        G1_num_of_nodes = data['number_of_nodes_in_G']
        not_found = len(data['S_perturbed_nodes']) - data['optimal_bitstring'].count('T')
        power = (not_found + 1) * np.log2(G1_num_of_nodes) + len(data['optimal_bitstring']) - 1
        probability_S_in_G1 = np.power(2, -power)
        eprint(f"probability of S in G1:{probability_S_in_G1}")
        eprint('not_found={}, power={}'.format(not_found, power))
        

    pval_bonf = pval2 / probability_S_in_G1
    eprint('p Bonferroni {}'.format(pval_bonf))
    res_dict = {'not_found': not_found, 'power': power, 'probability_S_in_G1': probability_S_in_G1, 
               'p_value_G1': pval1, 'p_value_G2': pval2, 'optimal_bs_1': data['optimal_bitstring'], 
                'optimal_bs_2': bs2, 'p_bonferroni': pval_bonf}

    if not_found < 0:  # Irregular sitation - notify user!
        out_result_path = 'flipped' + out_result_path
    json.dump(res_dict, open(out_result_path, 'w'))
    return pval_bonf

out_path = 'bonferroni_' + args.json_G1.split(os.path.sep)[-1]
calculate_bonferroni_pvalue(args.json_G1, args.json_G2, out_result_path=out_path)
exec_time = (time.time() - start_time)
eprint("--- %s seconds ---" % exec_time)