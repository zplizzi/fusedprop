import argparse
import torch
import sys
import pdb
import traceback


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="run name (optional)")
    parser.add_argument('--project',
                        default="grad_reversal_gan",
                        help="weights and biases project")
    parser.add_argument('--group',
                        default=None,
                        type=str,
                        help="weights and biases run group")

    parser.add_argument('--device',
                        default="cuda:0",
                        help="device (cpu or cuda:i)")
    parser.add_argument('--debug',
                        default=False,
                        type=str2bool,
                        help="drop into PDB on exception")

    # Adam parameters
    parser.add_argument('--lr',
                        default=.0002,
                        type=float,
                        help="learning rate")
    parser.add_argument('--lr_dis',
                        default=None,
                        type=float,
                        help="if set, use lr for G and lr_dis for D")
    parser.add_argument(
        '--n_dis',
        default=1,
        type=int,
        help="number of discriminator steps per generator step")
    parser.add_argument('--beta1', default=.5, type=float, help="adam beta1")
    parser.add_argument('--beta2', default=.999, type=float, help="adam beta2")

    parser.add_argument('--z_dim',
                        default=100,
                        type=int,
                        help="size of G input random variable")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--dataset',
                        default="cifar10",
                        choices=["cifar10", "mnist"])
    parser.add_argument('--model',
                        default="resnet",
                        choices=["mnist_fc", "sngan", "resnet"])
    parser.add_argument('--loss',
                        default="nonsaturating",
                        choices=[
                            "nonsaturating", "minimax", "hinge",
                            "least_squares", "wasserstein"
                        ])

    parser.add_argument(
        '--train_fn',
        default="baseline",
        type=str,
        choices=["baseline", "reversal"],
        help="use baseline for normal GAN training; reversal for FusedProp")
    parser.add_argument(
        '--lambda_inv',
        default=False,
        type=str2bool,
        help="enable lambda_inv mode when using FusedProp with hinge loss")
    # Whether to use SimGD instead of AltGD in the baseline train fn
    parser.add_argument(
        '--sim_gd',
        default=False,
        type=str2bool,
        help=
        "when using baseline train_fn, enable SimGD instead of the normal AltGD"
    )

    parser.add_argument('--spectral_norm',
                        default=False,
                        type=str2bool,
                        help="only applied to certain models")
    parser.add_argument('--batch_norm',
                        default=False,
                        type=str2bool,
                        help="only applied to certain models")

    parser.add_argument('--iterations',
                        default=None,
                        type=int,
                        help="number of G batches to train before termination")
    parser.add_argument('--evaluate_freq',
                        default=1000,
                        type=int,
                        help="run FID/IS computation every n G steps")
    parser.add_argument(
        '--log_freq',
        default=100,
        type=int,
        help="controls frequence of weights and biases metrics")
    parser.add_argument(
        '--data_root',
        default="./data",
        type=str,
        help="directory where datasets, FID stats, etc are stored")

    args = parser.parse_args(args)
    args.device = torch.device(args.device)
    return args


if __name__ == "__main__":
    local_args = parse_args(sys.argv[1:])

    # Do a late import since this takes some time and we want to get eg
    # argparse errors caught quickly.
    from gr_gan.trainable import Trainer
    trainer = Trainer(local_args)

    if trainer.args.debug:
        try:
            trainer.train()
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        trainer.train()
