from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--train_dir', type=str, help='Path to training data')
        self.parser.add_argument('--val_dir', type=str, help='Path to validation data')
        self.parser.add_argument('--exp_dir', type=str, help='Directory to store experiment results')

        self.parser.add_argument('--seed', default=42, type=int, help='Random seed for repeatability')

        ## Optimization parameters
        self.parser.add_argument(
            "--d_reg_every",
            type=int,
            default=16,
            help="interval of the applying r1 regularization",
        )
        self.parser.add_argument(
            "--g_reg_every",
            type=int,
            default=4,
            help="interval of the applying path length regularization",
        )
        self.parser.add_argument("--lr_g", type=float, default=0.001, help="learning rate")
        self.parser.add_argument("--lr_d", type=float, default=0.004, help="learning rate")

        ## Architecture parameters

    def parse(self):
        opts = self.parser.parse_args()
        return opts

