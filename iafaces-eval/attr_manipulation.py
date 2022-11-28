import os
import torch
from tqdm import tqdm
from prepare import build_model_and_dataset
from utils.eval_utils import make_sure_dir, save_img
import argparse
import numpy as np

# For calculating Table 4
# dicts = {"mouth_open": [[3], ["mouth"], 4],  # 977
#          "beard": [[3], ["mouth"], 9],
#          "bushy": [[0, 1], ['l_eye', 'r_eye'], 2.5],  #
#          "bags": [[0, 1], ['l_eye', 'r_eye'], 2.5],
#          "arched": [[0, 1], ['l_eye', 'r_eye'], 5],
#          "narrow_eyes": [[0, 1], ['l_eye', 'r_eye'], 2],
#          "male": [[0, 1, 2, 3], ['l_eye', 'r_eye', 'nose', 'mouth'], 2.5]
#          }

dicts = {"mouth_open": [[3], ["mouth"], 4],
         "beard": [[3], ["mouth"], 9],
         "bushy": [[0, 1], ['l_eye', 'r_eye'], 2.5],
         "bags": [[0, 1], ['l_eye', 'r_eye'], 2.5],
         "arched": [[0, 1], ['l_eye', 'r_eye'], 5],
         "narrow_eyes": [[0, 1], ['l_eye', 'r_eye'], 2],
         "male": [[0, 1, 2, 3], ['l_eye', 'r_eye', 'nose', 'mouth'], 2.5]
         }

if __name__ == '__main__':
    GPU = 0
    torch.cuda.set_device(GPU)
    device = torch.device(GPU)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default='iafaces-celebahq-256.pth', type=str)
    args.add_argument('--attr', default='mouth_open', type=str)
    args.add_argument('--data_path', default='data/attr_lists/mouth_close.txt', type=str)
    args.add_argument('--img_dir', default='data/CelebA-HQ-img', type=str)
    args = args.parse_args()

    expr_name = args.resume.split('.')[0]
    attr = args.attr
    index, component, range = dicts[attr]
    boundary_dir = os.path.join('output', expr_name, 'boundaries', attr)

    out_dir = os.path.join('output', expr_name, 'attr-manipulation', attr)
    make_sure_dir(out_dir)

    netE, netG, data_loader = build_model_and_dataset(args, device)
    #############################################
    # manipulations
    #############################################
    offset = []
    for i, name in enumerate(component):
        boundary = np.load(os.path.join(boundary_dir, '%s_boundary.npy' % name)).reshape([1, 1, -1])
        offset.append((range[i] if type(range) == list else range) * boundary)

    offset = np.concatenate(offset, axis=1)
    offset = torch.from_numpy(offset).cuda()

    with torch.no_grad():
        for batch_idx, (real_img, id) in enumerate(tqdm(data_loader)):
            real_img = real_img.cuda()
            out, gt_nodes = netE(real_img)

            gt_nodes[:, index, :] += offset[0]
            recon = netG(out, gt_nodes, randomize_noise=False)

            save_img(recon, os.path.join(out_dir, '%s.png' % str(id[0].numpy())))
