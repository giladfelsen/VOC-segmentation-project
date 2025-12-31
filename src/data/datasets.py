from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image   
import numpy as np
import albumentations as A  


class SegmentationDataset(Dataset):
 
    def __init__(self, image_dir: str, mask_dir: str, ids, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        #TODO add assertions and error handling

        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_id + '.png')
        

        image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L")) # assuming mask is grayscale
        mask = np.array(Image.open(mask_path)) # assuming mask is grayscale

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()  # ensure mask is long for loss calculation