## IA-FaceS — Official PyTorch Implementation

---

This repository contains  the **supplementary material** and  the **official PyTorch implementation** of the paper:<br />

**IA-FaceS: A Bidirectional Method for Semantic Face Editing**

> **Abstract:** *Semantic face editing has achieved substantial progress in recent years. However, existing face editing methods, which often encode the entire image into a single code, still have difficulty in enabling flexible editing while keeping high-fidelity reconstruction. The one-code scheme also brings entangled face manipulations and limited flexibility in editing face components. In this paper, we present IA-FaceS, a bidirectional method for disentangled face attribute manipulation as well as flexible, controllable component editing. We propose to embed images onto two branches: one branch computes high- dimensional component-invariant content embedding for capturing face details, and the other provides low-dimensional component-specific embeddings for component manipulations. The two-branch scheme naturally enables high-quality facial component-level editing while keeping faithful reconstruction with details. Moreover, we devise a component adaptive modulation (CAM) module, which integrates component- specific guidance into the decoder and successfully disentangles highly-correlated face components. The single-eye editing is developed for the first time without editing face masks or sketches. According to the experimental results, IA-FaceS establishes a good balance between maintaining image details and performing flexible face manipulation. Both quantitative and qualitative results indicate that the proposed method outperforms the existing methods in reconstruction, face attribute manipulation, and component transfer.*

## Demo video and supplementary file

---

Supplementary materials related to our paper are available at the following links:

| **Path**                                                     | **Description**                  |
| ------------------------------------------------------------ | -------------------------------- |
| [supplementary_material.pdf](https://drive.google.com/file/d/1fQTBCDFOWASF5awpqTBlu5iO4pgUVtoq/view?usp=sharing) | Supplementary file for IA-FaceS. |
| [IA-FaceS_demo.mp4](https://drive.google.com/file/d/1Rc6Licj_Trch7kWQhspOzozeJ7jbJgtH/view?usp=sharing) | The video demo of IA-FaceS.      |

## Installation

---

Install the dependencies:
```bash
conda create -n iafaces python=3.7
conda activate iafaces
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
For docker users:

```bash
docker pull huangwenjingcs/ubuntu18-conda-cuda11-pytorch1.7
```

## Datasets and pre-trained networks

---

To obtain the CelebA-HQ dataset, please refer to the [Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans). The official way of generating CelebA-HQ can be challenging. You can get the pre-generated dataset from [CelebA-HQ-dataset](https://drive.google.com/file/d/17wOT2Du1oKMU8DtRWupR_m1mgWvnGl1I/view?usp=sharing). Unzip the file and put the images to "data/CelebA-HQ-img/".

To obtain the  Flickr-Faces-HQ Dataset (FFHQ), please refer to [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset). Download the dataset and put the images to "data/ffhq-images1024/".

Pre-trained models can be found via the following links:

| Path                                                         | Description                                             |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| [checkpoint](https://drive.google.com/drive/folders/12XOUqeCVB8EDdU-d6JAszyzoKD7ms4qV?usp=share_link) | Main folder.                                            |
| ├  [iafaces-celebahq-256.pth](https://drive.google.com/file/d/1tHXOpMn7AGUYVmgDU-8oVRhcYims9jjZ/view?usp=share_link) | IA-FaceS trained with CelebA-HQ dataset at 256×256.     |
| ├  [iafaces_cam-celebahq-256.pth](https://drive.google.com/file/d/1Xm9juPMree52CdijllgbBqq_2I620VwE/view?usp=share_link) | IA-FaceS-CAM trained with CelebA-HQ dataset at 256×256. |
| ├  [iafaces-ffhq-1024.pth](https://drive.google.com/file/d/1DW6Ger9rSfHyn9mZabCWWfhhqfrdPJvY/view?usp=share_link) | IA-FaceS trained with FFHQ dataset at 1024×1024.        |
| ├  [iafaces_cam-ffhq-1024.pth](https://drive.google.com/file/d/1czzv-3fWqHUbFCMT2TLTdM1bf6zuZ8WY/view?usp=share_link) | IA-FaceS-CAM trained with FFHQ dataset at 1024×1024.    |

Download the pre-trained networks and put them to "checkpoints/".

## Train networks

---

Once the datasets are set up, you can train the networks as follows:

1. Edit `configs/<EXP_ID>.json` to specify the dataset, model and training configurations.
1. Run the training script with `python train.py -c configs/<EXP_ID>.json `. For example, 
```bash
 # train IA-FaceS with CelebA-HQ (256px) dataset, with a batch size of 16
python train.py -c configs/iafaces-celebahq-256.json --bz 16
```

   The code will use all GPUS by default, please specify the devices you want to use by:

```bash
 # train IA-FaceS in parallel, with a batch size of 16 (paper setting on celebahq)
CUDA_VISIBLE_DEVICES=0,1 python train.py -c configs/iafaces-celebahq-256.json --bz 8
```

3. The checkpoints are written to a newly created directory `saved/models/<EXP_ID>`

## Edit images 

---

For **attribute manipulation**, run:

```bash
python attr_manipulation.py --attr <ATTRIBUTE> --data_path <IMAGE_LIST> --resume <CHECKPOINT>
```

For **face component transfer**, run:

```bash
python component_transfer.py --component <COMPONENT> --target <TARGET_LIST> --reference <REFERENCE_LIST> --resume <CHECKPOINT>
```

For **image reconstruction**, run:

```bash
python reconstruction.py --data_path <IMAGE_LIST> --resume <CHECKPOINT>
```


The results are saved to `output/`.

## Metrics 

---

- Reconstruction:  MSE, LPIPS, PSNR,SSIM, FID
- Component transfer: MSE$_{\text{irr}}$, IFG, FID
- Attribute manipulation:  MSE$_{\text{irr}}$, IFG,Arc-dis

If you want to see details, please follow `evaluation/README.md`.

## Citation

---

If you find this work useful for your research, please cite our paper:
```
@article{huang2022ia,
  title={IA-FaceS: A bidirectional method for semantic face editing},
  author={Huang, Wenjing and Tu, Shikui and Xu, Lei},
  journal={Neural Networks},
  year={2022},
  publisher={Elsevier}
}
```

## Acknowledgement

---
This repository used some codes in [pytorch-template](https://github.com/victoresque/pytorch-template) and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).