import os
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import random
import albumentations as A
import cv2
from utils.data_utils import random_irregular_mask

BOX = np.array([[274, 360, 486, 550],
                [538, 360, 750, 550],
                [370, 530, 654, 700],
                [350, 690, 674, 870]])


class CelebAHQDataset(Dataset):
    def __init__(self, split, num_samples=None, data_dir=None,
                 image_dir=None,  image_size=(128, 128)):
        super(CelebAHQDataset, self).__init__()

        self.split = split

        self.image_ids = make_dataset(os.path.join(data_dir, '%s.txt' % split))
        self.image_size = image_size

        if num_samples is not None:
            random.shuffle(self.image_ids)
            self.image_ids = self.image_ids[:num_samples]

        self.image_dir = image_dir
        random.shuffle(self.image_ids)

        # define image transform
        transform = list()
        transform.append(A.Resize(height=image_size[0], width=image_size[1]))
        transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform)

    def __len__(self):
        num = len(self.image_ids)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (num_objs,)
        - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.image_dir, self.image_ids[index])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.split == 'train' and random.random() > 0.5:
            image = cv2.flip(image, 1)
        image = self.transform(image=image)['image']

        if self.split == 'train' and random.random() > 0.3:
            mask = random_irregular_mask(image, width=70)
        else:
            mask = torch.ones_like(image)
        return image, mask
