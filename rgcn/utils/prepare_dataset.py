from __future__ import print_function
import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

from rgcn.utils.data_utils import load_data
from rgcn.utils import *

import pickle as pkl

import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="aifb",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")

args = vars(ap.parse_args())

print(args)

# Define parameters
DATASET = args['dataset']

NUM_GC_LAYERS = 2  # Number of graph convolutional layers

# Get data
A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names = load_data(
    DATASET)

"""
At this point A is a list which is of a length double the number of relations, since each inverse relation is also
considered as a relation. Specifically, the adj mats with even index are the original relation adg mat, and the odd are
the corresponding inverse relation. (e.g. A[0] and A[1] describe the straight and inverse of the same relation.
"""

rel_list = list(range(len(A)))
for key, value in rel_dict.items():
    if value * 2 >= len(A):
        continue
    rel_list[value * 2] = key
    rel_list[value * 2 + 1] = key + '_INV'


num_nodes = A[0].shape[0]
A.append(sp.identity(A[0].shape[0]).tocsr())  # add identity matrix as last element in the list
# Maybe suppose to represent the 'self' relation?

support = len(A)

print("Relations used and their frequencies" + str([a.sum() for a in A]))

print("Calculating level sets...")

"""
Here they do some monkey business with the adj matrices.
"""

t = time.time()
# Get level sets (used for memory optimization)
# bfs_generator = bfs_relational(A, labeled_nodes_idx)
# lvls = list()
# lvls.append(set(labeled_nodes_idx))
# lvls.append(set.union(*next(bfs_generator)))
# print("Done! Elapsed time " + str(time.time() - t))
#
# # Delete unnecessary rows in adjacencies for memory efficiency
# todel = list(set(range(num_nodes)) - set.union(lvls[0], lvls[1]))
# for i in range(len(A)):
#     csr_zero_rows(A[i], todel)

data = {'A': A,
        'y': y,
        'train_idx': train_idx,
        'test_idx': test_idx
        }

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
