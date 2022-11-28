import os
import torch
from tqdm import tqdm
from prepare import build_model_and_dataset
from utils.eval_utils import make_sure_dir, save_img
import argparse

if __name__ == "__main__":
    GPU = 0
    torch.cuda.set_device(GPU)
    device = torch.device(GPU)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default='iafaces-celebahq-256.pth', type=str)
    args.add_argument('--data_path', default='data/celebahq-lists/code.txt', type=str)
    args.add_argument('--img_dir', default='data/CelebA-HQ-img', type=str)
    args = args.parse_args()

    netE, netG, data_loader = build_model_and_dataset(args, device)

    expr_name = args.resume.split('.')[0]
    expr_dir = 'output/{}/recon'.format(expr_name)
    out_dir = os.path.join(expr_dir, 'fake')
    real_dir = os.path.join(expr_dir, 'real')
    make_sure_dir(out_dir)
    make_sure_dir(real_dir)
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        real_img, img_id = batch
        real_img = real_img.cuda()

        out, nodes = netE(real_img)
        recon = netG(out, nodes, randomize_noise=False)

        save_img(recon, out_dir + '/%d.png' % img_id)
        save_img(real_img, real_dir + '/%d.png' % img_id)
