import os
import pandas as pd

result_dir = os.path.abspath('../../results/')


def compare_grids_with_and_without_features():
    with open(os.path.join(result_dir, 'first_grid_with_features.csv')) as features_res_file:
        features_res = pd.read_csv(features_res_file)
    with open(os.path.join(result_dir, 'first_grid_without_features.csv')) as original_res_file:
        original_res = pd.read_csv(original_res_file)

    print('hi')


if __name__ == '__main__':
    compare_grids_with_and_without_features()
