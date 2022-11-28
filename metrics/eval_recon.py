import argparse
from torchvision import transforms as T

import lpips
import torch.nn as nn
import torch
from utils.metric import psnr, ssim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils.dataset import CelebAHQDataset
from utils.tools import celebahq_depreprocess_batch


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, default=None, required=True)
    parser.add_argument("--fake", type=str, default=None, required=True)
    parser.add_argument("--list", type=str, default=None)
    parser.add_argument("-bz", "--batch_size", type=int, default=8, required=True)
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

    mse_results = []
    lpips_results = []
    total_psnr = []
    total_ssim = []
    mse_loss = nn.MSELoss(reduction='mean')
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    print(len(dataset))

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            real_img, fake_img = [img.cuda() for img in batch]

            mse = mse_loss(fake_img, real_img).item() * real_img.shape[0]
            lpips = percept(fake_img, real_img).sum().item()
            #
            mse_results.append(mse)
            lpips_results.append(lpips)

            real_img = celebahq_depreprocess_batch(real_img)
            fake_img = celebahq_depreprocess_batch(fake_img)

            total_psnr.append(psnr(real_img, fake_img) * real_img.shape[0])
            total_ssim.append(ssim(real_img, fake_img) * real_img.shape[0])

    mse_mean = sum(mse_results) / len(dataset)
    lpips_mean = sum(lpips_results) / len(dataset)
    psnr_mean = sum(total_psnr) / len(dataset)
    ssim_mean = sum(total_ssim) / len(dataset)

    print(mse_mean, lpips_mean, psnr_mean, ssim_mean)
