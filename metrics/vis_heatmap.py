import os
import argparse
from torchvision import transforms as T

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dataset import CelebAHQDataset
import cv2

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default="output/", type=str)
    parser.add_argument("--real", type=str, default=None, required=True)
    parser.add_argument("--fake", type=str, default=None, required=True)
    parser.add_argument("--list", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1, required=True)
    args = parser.parse_args()

    # images(0~1) are converted to -1 ~ 1
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CelebAHQDataset(args.real, args.fake, args.list, transform)
    init_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 2
    }
    data_loader = DataLoader(**init_kwargs)

    data = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            real_img, fake_img = [img.cuda() for img in batch]
            data += (abs(fake_img - real_img)).permute(0, 2, 3, 1).cpu().numpy().squeeze()

    os.makedirs(args.out_dir, exist_ok=True)
    data = data / len(dataset)
    data = (data - data.min()) / (data.max() - data.min()) * 255
    data = data.astype(np.uint8)
    w = cv2.applyColorMap(data, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.out_dir, 'heatmap.png'), w)
