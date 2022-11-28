import torch
import torch.nn.functional as F
from math import log10

import pytorch_ssim


def mse(output, target):
    with torch.no_grad():
        loss = F.mse_loss(output, target).item()
    return loss


def psnr(output, target):
    with torch.no_grad():
        total_psnr = 10 * log10(1 / mse(output, target))
    return total_psnr


def ssim(output, target):
    with torch.no_grad():
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        total_ssim = ssim_loss(output, target).item()
    return total_ssim


def tv_loss(image, target):
    with torch.no_grad():
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
               torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss.item()


def l1_loss(output, target):
    with torch.no_grad():
        loss = F.l1_loss(output, target).item()
    return loss



