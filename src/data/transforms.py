import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_baseline_train_transforms(img_size=256):
    """
    Minimal baseline — same for train/val/test except train is shuffled in loader.
    No geometric randomness. No color jitter.
    """

    return A.Compose([
        A.Resize(height=img_size, width=img_size),     # deterministic
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_basic_train_transforms(img_size=256):
    """
    Docstring for get_train_transforms
    
    :param img_size: Description
    """
    return A.Compose([
        A.SmallestMaxSize(max_size=img_size * 2, p=1.0),
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.2),
        
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])



def get_baseline_val_transforms(img_size=256):
    """
    Identical to train — ensures zero train/val distribution shift.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_baseline_test_transforms(img_size=256):
    """
    Identical to val — ensures zero val/test distribution shift.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])