import argparse

def base_arg_parser(*argument_providers):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dashboard', default=False, action='store_true')

    for provider in argument_providers:
        provider.add_parser_args(parser)

    optimizer_opts = parser.add_argument_group('optimizer')
    optimizer_opts.add_argument('--learning_rate', type=float, default=0.0001)
    optimizer_opts.add_argument('--weight_decay', type=float, default=0.0001)
    optimizer_opts.add_argument('--scheduler_step', type=int, default=100)
    optimizer_opts.add_argument('--epochs', type=int, default=800)

    checkpoint = parser.add_argument_group('checkpoint')
    checkpoint.add_argument('--checkpoint', default=False, action='store_true')
    checkpoint.add_argument('--checkpoint_epochs', type=int, default=5)

    return parser