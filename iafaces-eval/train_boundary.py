import os.path
import numpy as np
import os
import argparse

from utils.eval_utils import get_component, make_data, make_sure_dir
from utils.logger import setup_logger
from utils.manipulator import train_boundary


def find_boundary(args, nodes_pos, nodes_neg, name, out_dir, logger):
    """Main function."""

    min_len = min(len(nodes_neg), len(nodes_pos))
    nodes_neg = nodes_neg[0:min_len, :]
    nodes_pos = nodes_pos[0:min_len, :]
    label = np.concatenate([np.ones([min_len, 1]), np.zeros([min_len, 1])], axis=0)

    latent_codes = np.concatenate([nodes_pos[:min_len], nodes_neg[:min_len]], axis=0)
    latent_codes = latent_codes.reshape([latent_codes.shape[0], -1])

    boundary, acc = train_boundary(latent_codes=latent_codes,
                                   scores=label,
                                   chosen_num_or_ratio=args.chosen_num_or_ratio,
                                   split_ratio=args.split_ratio,
                                   invalid_value=args.invalid_value,
                                   logger=logger)

    np.save(os.path.join(out_dir, '%s_boundary.npy' % name), boundary)

    return acc


def main(args):
    """Main function."""

    out_dir = os.path.join('output', args.expr, 'boundaries', args.pos_attr)
    make_sure_dir(out_dir)
    component = get_component(args.index)
    nodes_path = os.path.join('output/%s/codes/latent.pkl' % args.expr)

    nodes_pos = make_data(nodes_path,
                          os.path.join('data/boundary_lists/', args.pos_attr + '.txt'),
                          args.index)

    nodes_neg = make_data(nodes_path,
                          os.path.join('data/boundary_lists/', args.neg_attr + '.txt'),
                          args.index)

    logger = setup_logger(out_dir, logger_name='generate_data')

    logger.info(f'Training classifier according to %s:' % component)

    r_acc = find_boundary(args, nodes_pos, nodes_neg, component, out_dir, logger)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--expr', default='iafaces-celebahq-256', type=str)
    args.add_argument('--index', default=0, type=int)
    args.add_argument('--pos_attr', default='big_nose', type=str)
    args.add_argument('--neg_attr', default='no_big_nose', type=str)
    args.add_argument('--chosen_num_or_ratio', type=float, default=0.47,
                      help='How many samples to choose for training. '
                           '(default: 0.2)')
    args.add_argument('--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
    args.add_argument('--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')
    args = args.parse_args()
    main(args)
