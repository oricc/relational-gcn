from rgcn.models.BaseRGCN import BasicRGCN
import itertools


class GridRGCNModel(BasicRGCN):
    """
    This class is an example of how to run grids over multiple configurations.
        - The train method is overridden so that it changes the args at every iteration
        - The build model method is called at every iteration
        - The data method is still called only once, saving time
    """

    def _generate_config_list(self):
        """
        This is the new method for the grid model.
        This method is responsible for generating the grid configurations.
        This method is built as a generator for memory's sake, but if overridden, could also
            return a list of configuration.

        A configuration is a dictionary with the following keys:
            -'dataset' - the name of the dataset file to use
            -'epochs' - the number of epochs to train for
            -'validation' - whether we are using validation or test data
            -'learnrate' - the learning rate for the optimizer
            -'l2norm' - the weight decay factor for the regularization
            -'hidden' - the number of hidden nodes (the output dimension of the hidden layer)
            -'bases' - the number of bases used (for more information on bases, see the original paper)
            -'dropout' - the amount of dropout to use
        :return: a generator of configurations
        """
        dataset_options = ['aifb', 'mutag']
        epochs_options = [50]
        validation_options = [True]
        lr_options = [0.01]
        l2_options = [0.001, 0.01, 0.1, 1]
        hidden_options = [8, 16, 32, 64]
        bases_options = [4, 8, 16, 32]
        dropout_options = [0.2, 0.5, 0.8, 0.9]

        for d, e, v, lr, l2, h, b, do in itertools.product(dataset_options, epochs_options, validation_options,
                                                           lr_options, l2_options, hidden_options, bases_options,
                                                           dropout_options):
            config = {
                'dataset': d,
                'epochs': e,
                'validation': v,
                'learnrate': lr,
                'l2norm': l2,
                'hidden': h,
                'bases': b,
                'dropout': do
            }
            yield config

    def train(self):
        for config in self._generate_config_list():
            self._set_args(config)
            self._build_model()
            super().train()
