from __future__ import print_function

import os
import sys
import pickle as pkl
import time
import tensorflow as tf

import networkx as nx
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

from features_infra.feature_calculators import FeatureMeta
from features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from features_infra.graph_features import GraphFeatures

from rgcn.layers.graph import GraphConvolution, AsymmetricGraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *

from rgcn.models.BaseRGCN import BasicRGCN


class ASymmetricRGCN(BasicRGCN):

    @staticmethod
    def __matrix_sym_normalization(mx):
        rowsum = np.array(mx.sum(1))
        rowsum[rowsum != 0] **= -0.5
        r_inv = rowsum.flatten()
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx).dot(r_mat_inv)  # D^-0.5 * X * D^-0.5

    def __asymmetric_normalization(self):
        A = []
        for i in range(0, self.support - 1, 2):
            mx = self.A[i] + sp.eye(self.A[i].shape[0])
            mx_t = mx.transpose()

            mx = ASymmetricRGCN.__matrix_sym_normalization(mx)
            mx_t = ASymmetricRGCN.__matrix_sym_normalization(mx_t)

            A.extend([mx, mx_t])
        A.append(self.A[-1])
        self.A = A

    def _get_data(self):
        self._normalizing_function = self.__asymmetric_normalization
        super()._get_data()

    def _build_model(self):
        A_in = [InputAdj(sparse=True) for _ in range(self.support)]
        # A_in = [InputAdj(sparse=True) for _ in range(support)]
        X_in = Input(shape=(self.X.shape[1],), sparse=True)
        # X_in = Input(shape=(X.shape[1],), sparse=True)

        # Define model architecture
        H = AsymmetricGraphConvolution(self.HIDDEN, self.support, num_bases=self.BASES, featureless=self.featureless,
                                       activation='relu', bias=True,
                                       W_regularizer=l2(self.L2))([X_in] + A_in)
        H = Dropout(self.DO)(H)
        Y = AsymmetricGraphConvolution(self.train_labels.shape[1], self.support, num_bases=self.BASES, bias=True,
                                       activation='softmax')([H] + A_in)
        # H = Dropout(self.DO)(H)
        # Y = Dense(self.train_labels.shape[1], activation='softmax', kernel_regularizer=l2(self.L2))(H)

        # Compile model
        model = Model(inputs=[X_in] + A_in, outputs=Y)
        # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.LR, momentum=0.99))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LR))

        return model


class AsymmetricRGCNWithNeighborHistograms(ASymmetricRGCN):

    def __init__(self, args):
        super().__init__(args)
        self.featureless = False

    @staticmethod
    def __sum_sparse(m):
        x = np.zeros(m[0].shape)
        for a in m:
            ri = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
            x[ri, a.indices] += a.data
        return sp.csr_matrix(x)

    def __get_features(self):
        with open(self.dirname + '/data/' + self.DATASET + '.pickle', 'rb') as f:
            data = pkl.load(f)

        self._original_adj_matrices = data['A']
        y = data['y']

        node_labels = []
        for a in y.todense():
            if a.max() != 0:
                node_labels.append(a.argmax())
            else:
                node_labels.append(-1)

        sum_adj = AsymmetricRGCNWithNeighborHistograms.__sum_sparse(self.A)
        gnx = nx.from_scipy_sparse_matrix(sum_adj, parallel_edges=True)
        gnx = nx.DiGraph(gnx, labels=node_labels)

        for n, label in zip(gnx.nodes, node_labels):
            gnx.node[n]['label'] = label

        real_labels = list(set(node_labels) - {-1})

        # Get the features for the graph
        NEIGHBOR_FEATURES = {
            "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1, labels_to_consider=real_labels),
                                                    {"fnh", "first_neighbor"}),
            "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2, labels_to_consider=real_labels),
                                                     {"snh", "second_neighbor"}),
        }
        features_path = os.path.join(os.path.abspath('../features'), self.DATASET)
        features = GraphFeatures(gnx, NEIGHBOR_FEATURES, dir_path=features_path)
        features.build(include=set(self.idx_train), should_dump=True)

        add_ones = bool({"first_neighbor_histogram", "second_neighbor_histogram"}.intersection(NEIGHBOR_FEATURES))
        _topo_mx = features.to_matrix(add_ones=add_ones, dtype=np.float64, mtype=np.matrix, should_zscore=True)

        del data
        return sp.csr_matrix(
            np.hstack([_topo_mx, np.zeros((_topo_mx.shape[0], _topo_mx.shape[0] - _topo_mx.shape[1]))]))

    def _get_data(self):
        super()._get_data()
        self.X = self.__get_features()
        print("done feature")
