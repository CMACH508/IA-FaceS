import os
import torch
from tqdm import tqdm
from prepare import build_model_and_dataset
from utils.eval_utils import make_sure_dir
import argparse
import pickle


def save_pkl(table, data_dir, name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(table, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    GPU = 0
    torch.cuda.set_device(GPU)
    device = torch.device(GPU)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default='iafaces-celebahq-256.pt', type=str)
    args.add_argument('--data_path', default='data/celebahq-lists/code.txt', type=str)
    args.add_argument('--img_dir', default='data/CelebA-HQ-img', type=str)
    args = args.parse_args()

    netE, netG, data_loader = build_model_and_dataset(args, device)

    expr_name = args.resume.split('.')[0]
    out_dir = 'output/{}/codes'.format(expr_name)
    make_sure_dir(out_dir)
    codes = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            real_img, img_id = batch
            real_img = real_img.cuda()
            img_id = img_id[0].numpy()
            out, nodes = netE(real_img)
            nodes = nodes.cpu().numpy()
            codes[str(img_id)] = nodes
    save_pkl(codes, out_dir, 'latent')
