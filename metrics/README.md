## Metrics

- Reconstruction: MSE, LPIPS, PSNR,SSIM, FID
- Component transfer: MSE$_{\text{irr}}$, IFG, FID
- Attribute manipulation: Accuracy, MSE$_{\text{irr}}$, IFG, Arc-dis

All command lines should be run in `IA-FaceS/evaluation`

**MSE, LPIPS, PSNR,SSIM**

```bash
python eval_recon.py --real <REAL FACES> --fake <RECONSTRUCTIONS> -bz <BATCH SIZE> 
```

**MSE$_{\text{irr}}$, IFG**

First, download the pre-trained StyleGAN2 discriminator from [stylegan2.pt](https://drive.google.com/file/d/1boSiIuC4qiCGcqy58svconr1MKVenn8t/view?usp=sharing) and put it to `checkpoint/latest.pt`

```bash
python d_score.py --component <COMPONENET> --src <RECONSTRUCTIONS> --edit <EDITED FACES>
# <COMPONENET> is used to define the component-irrelevant regions for calculating MSE$_{\text{irr}}$
```

**Accuracy**

First, download the pre-trained attribute classifiers from [stylegan2.pt](https://drive.google.com/file/d/1boSiIuC4qiCGcqy58svconr1MKVenn8t/view?usp=sharing) and put them to `classifiers/`.

Second, install the requirements following [stylegan2](https://github.com/NVlabs/stylegan2).

Finally, run:

```bash
python attr_acc.py --fake <EDITED FACES> --attr_idx <ATTRIBUTE INDEX>
```

**Arc-dis**

First, install deepface as:

```bash
pip install deepface
```

Second, run:

```bash
python arc_dis.py --fake <EDITED FACES> --attr <ATTRIBUTE> --real <RECONSTRUCTIONS> --method <MODEL NAME>
```

**FID**

```bash
python -m pytorch_fid <REAl FACES> <EDITED FACES> # returns FID
```



