import os
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import numpy as np

BOX = np.array([[274, 360, 486, 550],
                [538, 360, 750, 550],
                [370, 530, 654, 700],
                [350, 690, 674, 870]])


class CelebAHQDataset(Dataset):
    def __init__(self, data_path=None, image_dir=None, image_size=256):
        super(CelebAHQDataset, self).__init__()

        self.image_dir = image_dir

        self.image_ids = make_dataset(data_path)

        self.image_size = image_size

        # define image transform
        transform = list()
        transform.append(A.Resize(height=image_size, width=image_size))
        transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform.append(ToTensorV2())
        self.image_transform = A.Compose(transform)
        self.file_type = ('.png' if image_size == 1024 else '.jpg')
        # self.file_type = '.jpg'

    def __len__(self):
        num = len(self.image_ids)
        return num

    def __getitem__(self, index):
        name = self.image_ids[index]
        if self.file_type not in name:
            name = str(int(name)) + self.file_type
        img_path = os.path.join(self.image_dir, name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)['image']
        return image, int(self.image_ids[index].split('.')[0])
