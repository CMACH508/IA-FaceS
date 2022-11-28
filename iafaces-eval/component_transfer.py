from utils.eval_utils import make_sure_dir, make_dataset_txt, save_img
import tqdm
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import cv2
from prepare import build_model_and_dataset
import argparse

transform = list()
transform.append(A.Resize(height=256, width=256))
transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform.append(ToTensorV2())
image_transform = A.Compose(transform)


def preprocess_img(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    in_image = image_transform(image=image)['image'].unsqueeze(0)
    return in_image


name_to_index = {
    'eyes': [0, 1],
    'nose': 2,
    'mouth': 3,
    'l_eye': 0,
    'r_eye': 1
}

if __name__ == "__main__":
    GPU = 0
    torch.cuda.set_device(GPU)
    device = torch.device(GPU)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--resume', default='iafaces-celebahq-256.pth', type=str)
    args.add_argument('--target', default='data/celebahq-lists/test.txt', type=str)
    args.add_argument('--component', default='eyes', type=str)
    args.add_argument('--reference', default='data/celebahq-lists/shuffled_test.txt', type=str)
    args.add_argument('--img_dir', default='data/CelebA-HQ-img', type=str)
    args = args.parse_args()

    netE, netG, _ = build_model_and_dataset(args, device, only_model=True)
    expr_name = args.resume.split('.')[0]
    out_dir = 'output/{}/component_transfer/{}'.format(expr_name, args.component)

    make_sure_dir(out_dir)
    sources = make_dataset_txt(args.target)
    references = make_dataset_txt(args.reference)

    index = name_to_index[args.component]

    with torch.no_grad():
        for idx, src_filename in enumerate(tqdm.tqdm(sources)):
            ref_image = preprocess_img(os.path.join(args.img_dir, references[idx]))
            ref_face, ref_nodes = netE(ref_image.cuda())

            src_image = preprocess_img(os.path.join(args.img_dir, src_filename))

            out, nodes = netE(src_image.cuda())

            nodes[:, index, :] = ref_nodes[:, index, :]
            recon = netG(out, nodes, randomize_noise=False)

            save_img(recon, os.path.join(out_dir, src_filename.split('.')[0] + '.png'))
