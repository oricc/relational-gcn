from __future__ import print_function

import os
import sys
import pickle as pkl
import time
import tensorflow as tf
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *

VERBOSE = False


class RGCNModel:

    def __init__(self, original=None, **kwargs):
        # The model object to train
        self.model = None

        # The data to use as input to the model
        self.train_data = None
        self.test_data = None
        self.validation_data = None

        # The labels for each of the sets
        self.train_labels = None
        self.test_labels = None
        self.validation_labels = None

        # The masks for each of the sets
        self.idx_train, self.idx_test, self.idx_val = None, None, None

        # The feature matrix used as input
        self.X = None

        # The list of adjacency matrices used as input
        self.A = None

        if not original:
            # Set all the data variables defined above
            self._get_data()
        else:
            # Copy the data from the original instance
            self._copy_data_from(original)

        # Build the model structure
        self.model = self._build_model()

    def train(self):
        """
        This function is responsible for the actual training of the model.
        The canonical training loop is located in the BasicRGCN class.
        """
        raise NotImplementedError()

    def _get_data(self):
        """
        This method is responsible for setting all the variables used in the training proccess,
        both those defined in this class' constructor and those defined in the constructor of the
        inheriting class.
        """
        raise NotImplementedError()

    def _copy_data_from(self, original):
        """
        This method is essentially used as a copy constructor, used to initialize the data from an
        instance that is already created
        :param original: the original instance
        """
        raise NotImplementedError()

    def _build_model(self):
        """
        This model build the model to train.
        This method runs only after get_data, and so all the variables are initialized.
        :return: a model
        """
        raise NotImplementedError()

    def clear(self):
        del self.model
        self.model = None

        del self.A
        self.A = None

        del self.X
        self.X = None


class BasicRGCN(RGCNModel):
    """
    This is the base class used for training an RGCN. The class contains the standard data loading and training
    code that can be used with any model, as well as the argument loading for grids.

    """

    def __init__(self, args=None, original=None, **kwargs):
        # Define parameters
        if args:
            self._set_args(args)
        self.dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

        self.train_mask = None
        self.num_nodes = None
        self.support = None
        self._normalizing_function = self.__symmetric_normalization

        self.featureless = True
        kwargs['build_data_and_model'] = kwargs.get('build_data_and_model', True) and (args is not None)
        super(BasicRGCN,self).__init__(original=original, **kwargs)

    def _set_args(self, args):
        self.DATASET = args['dataset']
        self.NB_EPOCH = args['epochs']
        self.VALIDATION = args['validation']
        self.LR = args['learnrate']
        self.L2 = args['l2norm']
        self.HIDDEN = args['hidden']
        self.BASES = args['bases']
        self.DO = args['dropout']

    def __symmetric_normalization(self):

        A = self.A
        # Normalize adjacency matrices individually
        for i in range(len(A)):
            d = np.array(A[i].sum(1)).flatten()
            d_inv = 1. / d
            d_inv[np.isinf(d_inv)] = 0.
            D_inv = sp.diags(d_inv)
            A[i] = D_inv.dot(A[i]).tocsr()

    def _get_data(self):

        with open(self.dirname + '/data/' + self.DATASET + '.pickle', 'rb') as f:
            data = pkl.load(f)

        A = data['A']
        y = data['y']
        train_idx = data['train_idx']
        test_idx = data['test_idx']

        # Get dataset splits
        self.train_labels, self.validation_labels, \
        self.test_labels, self.idx_train, self.idx_val, self.idx_test = get_splits(y, train_idx,
                                                                                   test_idx,
                                                                                   self.VALIDATION)
        self.train_mask = sample_mask(self.idx_train, y.shape[0])

        self.num_nodes = A[0].shape[0]
        self.support = len(A)
        # Define empty dummy feature matrix (input is ignored as we set featureless=True)
        # In case features are available, define them here and set featureless=False.
        self.X = sp.csr_matrix(A[0].shape)

        self.A = A

        self._normalizing_function()

    def _copy_data_from(self, original):
        # Get dataset splits
        self.train_labels, self.validation_labels, \
        self.test_labels, self.idx_train, self.idx_val, self.idx_test = \
            original.train_labels, original.validation_labels, \
            original.test_labels, original.idx_train, original.idx_val, original.idx_test
        self.train_mask = original.train_mask
        self.A = original.A
        self.num_nodes = self.A[0].shape[0]
        self.support = len(self.A)
        # Define empty dummy feature matrix (input is ignored as we set featureless=True)
        # In case features are available, define them here and set featureless=False.
        self.X = original.X

    def _build_model(self):

        A_in = [InputAdj(sparse=True) for _ in range(self.support)]
        # A_in = [InputAdj(sparse=True) for _ in range(support)]
        X_in = Input(shape=(self.X.shape[1],), sparse=True)
        # X_in = Input(shape=(X.shape[1],), sparse=True)

        # Define model architecture
        H = GraphConvolution(self.HIDDEN, self.support, num_bases=self.BASES, featureless=self.featureless,
                             activation='relu',
                             W_regularizer=l2(self.L2))([X_in] + A_in)
        H = Dropout(self.DO)(H)
        Y = GraphConvolution(self.train_labels.shape[1], self.support, num_bases=self.BASES,
                             activation='softmax')([H] + A_in)
        # H = Dropout(self.DO)(H)
        # Y = Dense(self.train_labels.shape[1], activation='softmax', kernel_regularizer=l2(self.L2))(H)

        # Compile model
        model = Model(inputs=[X_in] + A_in, outputs=Y)
        # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.LR, momentum=0.99))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LR))

        return model

    def train(self):
        # Fit
        preds = None
        for epoch in range(1, self.NB_EPOCH + 1):

            # Log wall-clock time
            t = time.time()

            # Single training iteration
            self.model.fit([self.X] + self.A, self.train_labels, sample_weight=self.train_mask,
                           batch_size=self.num_nodes, epochs=1, shuffle=False, verbose=0)

            if epoch % 1 == 0:

                # Predict on full dataset
                preds = self.model.predict([self.X] + self.A, batch_size=self.num_nodes)

                # Train / validation scores
                train_val_loss, train_val_acc = evaluate_preds(preds, [self.train_labels, self.validation_labels],
                                                               [self.idx_train, self.idx_val])

                if VERBOSE:
                    print("Epoch: {:04d}".format(epoch),
                          "train_loss= {:.4f}".format(train_val_loss[0]),
                          "train_acc= {:.4f}".format(train_val_acc[0]),
                          "val_loss= {:.4f}".format(train_val_loss[1]),
                          "val_acc= {:.4f}".format(train_val_acc[1]),
                          "time= {:.4f}".format(time.time() - t))

            else:
                if VERBOSE:
                    print("Epoch: {:04d}".format(epoch),
                          "time= {:.4f}".format(time.time() - t))

        # Testing
        test_loss, test_acc = evaluate_preds(preds, [self.test_labels], [self.idx_test])
        if VERBOSE:
            print("Test set results:",
                  "loss= {:.4f}".format(test_loss[0]),
                  "accuracy= {:.4f}".format(test_acc[0]))
