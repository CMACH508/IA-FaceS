import torch
import argparse
import collections
import torch.multiprocessing as mp

from trainer.trainer import Trainer
from parse_config import ConfigParser

from core.philly import ompi_size, ompi_local_size, ompi_rank, ompi_local_rank
from core.philly import get_master_ip, ompi_universe_size
from core.utils import set_seed
import warnings


def main_worker(gpu, config):
    if 'local_rank' not in config._config:
        config.__add_item__('local_rank', gpu)
        config.__add_item__('global_rank', gpu)

    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        print('using GPU {} for training'.format(int(config['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch'
                                             )
    set_seed(config['seed'])
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/iafaces-cam-celebahq.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--run_id', default=None, type=str,
                      help='run id (default: all)')

    args.add_argument('-p', '--port', default='1234', type=str,
                      help='the port')
    args.add_argument('-s', '--seed', default='123', type=str,
                      help='the port')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bz', '--batch_size'], type=int, target='data_loader;batch_size'),
        CustomArgs(['--d', '--data_name'], type=int, target='data_name'),
        CustomArgs(['--m', '--model_arch'], type=str, target='model_arch'),
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print('check if the gpu resource is well arranged on philly')
    assert ompi_size() == ompi_local_size() * ompi_universe_size()
    # setup distributed parallel training environments
    world_size = ompi_size()
    ngpus_per_node = torch.cuda.device_count()
    if world_size > 1:
        config.__add_item__('world_size', world_size)
        config.__add_item__('init_method', 'tcp://' + get_master_ip() + args.port)
        config.__add_item__('distributed', True)
        config.__add_item__('local_rank', ompi_local_rank())
        config.__add_item__('global_rank', ompi_rank())
        main_worker(0, config, )
    elif ngpus_per_node > 1:
        config.__add_item__('world_size', ngpus_per_node)
        config.__add_item__('init_method', 'tcp://127.0.0.1:' + args.port)
        config.__add_item__('distributed', True)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(config,))
    else:
        config.__add_item__('world_size', 1)
        config.__add_item__('distributed', False)
        main_worker(0, config, )
