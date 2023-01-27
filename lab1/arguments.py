
import argparse

import torch
def get_args(arg_list=None):
    parser = argparse.ArgumentParser()

    # used for auto_train.py to split training data
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-train_ratios', type=float, nargs='+', default=[0.8, 0.5, 0.2],
                        help='proportion of data used for training (default 0.6)')
    parser.add_argument('-dev_ratios', type=float, nargs='+', default=[0.1, 0.25, 0.4],
                        help='proportion of data used for validation (default 0.2)')
    parser.add_argument('-test_ratio', type=float, default=0,
                        help='proportion of data used for testing (default 1 - train_ratio - dev_ratio)')

    parser.add_argument('-epoch', type=int, default=10000)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-patience', type=int, default=100)
    parser.add_argument('-seeds', nargs='+', default=[i for i in range(100)], help='seeds')

    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args
