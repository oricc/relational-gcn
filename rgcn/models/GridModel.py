import itertools
import os
from utils.utils import evaluate_preds


class GridRunner:
    RESULTS_DIR = 'results'
    """
    This class is an example of how to run grids over multiple configurations.
        - The train method is overridden so that it changes the args at every iteration
        - The build model method is called at every iteration
        - The data method is still called only once, saving time
    """

    def __init__(self, result_file_name, rgcn_model):
        if '.csv' not in result_file_name:
            result_file_name = result_file_name + '.csv'
        self.file_name = os.path.join(os.path.dirname(os.getcwd()), GridRunner.RESULTS_DIR, result_file_name)
        self.rgcn_model = rgcn_model

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
        validation_options = [False]
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

    def run(self):
        current_dataset = None
        for config in self._generate_config_list():
            print(config)
            self.rgcn_model._set_args(config)
            if config['dataset'] != current_dataset:
                # Make sure to only load the dataset when it changes
                current_dataset = config['dataset']
                self.rgcn_model._get_data()
            self.rgcn_model.model = self.rgcn_model._build_model()
            self.rgcn_model.train()
            self._eval_model(config)

    def _eval_model(self, config):
        evals = config.copy()

        # Predict on full dataset
        preds = self.rgcn_model.model.predict([self.rgcn_model.X] + self.rgcn_model.A,
                                              batch_size=self.rgcn_model.num_nodes)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [self.rgcn_model.train_labels,
                                                               self.rgcn_model.validation_labels],
                                                       [self.rgcn_model.idx_train, self.rgcn_model.idx_val])
        evals.update({
            'train_loss': train_val_loss[0],
            'train_acc': train_val_acc[0],
            'val_loss': train_val_loss[1],
            'val_acc': train_val_acc[1]

        })

        # Testing

        test_loss, test_acc = evaluate_preds(preds, [self.rgcn_model.test_labels], [self.rgcn_model.idx_test])
        evals.update({
            'test_loss': test_loss[0],
            'test_acc': test_acc[0]
        })

        self._save_to_file(evals)

    def _save_to_file(self, evals):
        if not os.path.exists(self.file_name):
            # If the file doesn't exist, we need to both create and initialize it
            self._init_save_file(evals)

        # Save the results
        with open(self.file_name, 'w') as f:
            f.write(','.join(evals) + '\n')

    def _init_save_file(self, evals):
        """
        This method creates and add a header line for the csv results file.
        The header is a list of the keys in the evaluation dictionary, which are assumed to stay
        the same for the duration of the grid.
        :param evals: the dictionary containing both the configuration and the evaluation results
        """
        with open(self.file_name, 'w+') as f:
            f.write(','.join(evals.keys()) + '\n')
