import os
from utils.model import Generator, Discriminator
import pickle
from torchvision.utils import save_image, make_grid
from torchvision import transforms as T
import torch


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths


def save_img(img, path):
    if type(img) is list:
        img = make_grid(torch.cat(img, dim=0), nrow=len(img), padding=0, normalize=True, range=(-1, 1))
        save_image(img, path)
    else:
        save_image(img, path, normalize=True, padding=0, range=(-1, 1))
    return


def load_model(args, device):
    g_ema = Generator(args.img_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    return g_ema


def load_d_model(args, device):
    d_ema = Discriminator(args.img_size, 2)
    d_ema.load_state_dict(torch.load(args.ckpt)["d"], strict=False)
    d_ema.eval()
    d_ema = d_ema.to(device)
    return d_ema


def save_pkl(table, data_dir, name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(table, f, pickle.HIGHEST_PROTOCOL)


def make_sure_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def celebahq_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=[2.0, 2.0, 2.0]),
        T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def celebahq_depreprocess_batch(imgs, rescale=True):
    """
        Input:
        - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

        Output:
        - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
          in the range [0, 255]
        """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = celebahq_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        netDe = deprocess_fn(imgs[i])[None]
        # netDe = netDe.mul(255).clamp(0, 255).byte()
        imgs_de.append(netDe)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de
