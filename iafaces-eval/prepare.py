import os
import torch
from importlib import import_module
from torch.utils.data import DataLoader
from data_loader.celebahq import CelebAHQDataset


def build_model_and_dataset(args, device, only_model=False):
    checkpoint = "checkpoints/{}".format(args.resume)
    if not os.path.isfile(checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % checkpoint)
        return
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(checkpoint, map_location=map_location)
    config = checkpoint['config']
    arch = config['model_arch']
    model_arch = import_module('model.' + arch)

    netE = model_arch.Encoder(**config['encoder']['args'])
    netG = model_arch.Generator(**config['generator']['args'])
    netE.load_state_dict(checkpoint['e_ema'])
    netG.load_state_dict(checkpoint['g_ema'])

    netE.eval()
    netG.eval()
    netE.to(device)
    netG.to(device)

    #########################################
    # build data loader
    #########################################
    if not only_model:
        init_kwargs = {
            "data_path": args.data_path,
            "image_dir": args.img_dir,
            "image_size": config['data_set']['args']['image_size'][0]
        }
        val_dataset = CelebAHQDataset(**init_kwargs)
        init_kwargs = {
            'dataset': val_dataset,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1
        }
        data_loader = DataLoader(**init_kwargs)
    else:
        data_loader = None
    return netE, netG, data_loader
