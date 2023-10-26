import torch.nn.init as init

from argparse import ArgumentParser


def parse_arguments(argv):
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default: 42')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='MLP', help='Model name, default: MLP')

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default: 32')
    
    # Optimizer hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs, default: 10')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer, default: Adam')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate, default: 1e-3')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay, default: 0')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum, default: 0')

    # MLP hyperparameters
    parser.add_argument('--hidden_layer', type=int, nargs='+', default=[256], help='Hidden layer size, default: [256]')


    return parser.parse_args(argv)


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        # count = v.data.numel()
        # v = v.data.sum()
        self.n_count += 1#count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
    

def initialize_weights(model):
    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue