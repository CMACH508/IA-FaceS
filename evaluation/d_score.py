import os
import torch
import argparse
from utils.tools import load_d_model
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import tqdm
import torch.nn.functional as F
import numpy as np

GPU = 0
torch.cuda.set_device(GPU)
device = torch.device(GPU)

transform = list()

transform.append(A.Resize(height=256, width=256))
transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform.append(ToTensorV2())
transform = A.Compose(transform)


def get_node_box(fs, ori_size, boxes):
    scale = ori_size / fs
    rescaled_boxes = torch.ceil((boxes - scale / 2) / scale).int()
    rescaled_boxes = torch.clamp(rescaled_boxes, 0, fs - 1).to(torch.int32)
    return rescaled_boxes


def process_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = image.unsqueeze(0).cuda()
    return image


name_to_index = {
    'eyes': [0, 1],
    'nose': [2],
    'mouth': [3],
    'l_eye': [0],
    'r_eye': [1]
}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    d_ema = load_d_model(args, device)

    src_dir = args.src
    edited = args.edit

    box = torch.from_numpy(np.array([[68, 90, 121, 137],
                                     [134, 90, 187, 137],
                                     [92, 132, 163, 175],
                                     [87, 172, 168, 217]]))

    score1 = 0
    score2 = 0

    mean_mse = 0
    img_lists = list(os.listdir(edited))

    mask = torch.ones([256, 256]).cuda()
    index = name_to_index[args.component]

    for i in index:
        print(i)
        mask[box[i][1]:box[i][3], box[i][0]:box[i][2]] = 0

    with torch.no_grad():
        for batch_idx, filename in enumerate(tqdm.tqdm(img_lists)):
            # print(os.path.join(src_dir, filename))
            img1 = process_img(os.path.join(src_dir, filename))
            img2 = process_img(os.path.join(edited, filename))

            mean_mse += F.mse_loss(img1 * mask, img2 * mask).item()
            score1 += d_ema(img1).item()
            score2 += d_ema(img2).item()

        print(mean_mse / len(img_lists))
        score_drop = score2 / len(img_lists) - score1 / len(img_lists)
        print(score_drop)

    with open(os.path.join(args.out_dir, 'mse_score.txt'), 'a') as f:
        f.write('%f' % (mean_mse / len(img_lists)))
        f.write('\n')

    with open(os.path.join(args.out_dir, 'd_score_drop.txt'), 'a') as f:
        f.write('%f' % score_drop)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train semantic boundary with given latent codes and '
                    'attribute scores.')

    parser.add_argument('--src', type=str)
    parser.add_argument('--edit', type=str)
    parser.add_argument('--component', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--ckpt', default='checkpoint/latest.pt', type=str)
    parser.add_argument('--out_dir', default="output/", type=str)
    args = parser.parse_args()

    main(args)
