

from pathlib import Path
import os
import test
import yaml
import random
from torch.utils.data import DataLoader

from .datasets import SegmentationDataset
from .transforms import get_baseline_train_transforms, get_baseline_val_transforms, get_baseline_test_transforms



# ---- config ----
def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---- project root ----
def project_root():
    return Path(__file__).resolve().parents[2]


# ---- dataset root ----
def dataset_root(cfg):
    # allow env override (e.g., Colab/Kaggle)
    base = os.getenv("DATA_DIR", cfg["data_dir"])
    return project_root() / base / cfg["dataset_name"]


# ---- utils ----
def read_ids(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def split_val_ids(val_ids, frac, seed=1337):
    val_ids = list(val_ids)
    random.Random(seed).shuffle(val_ids)
    k = int(len(val_ids) * frac)
    return val_ids[:k], val_ids[k:]


def maybe_truncate(ids, n):
    return ids if n is None else ids[:n]


# ---- dataset construction ----
def get_datasets(cfg, train_tfms=None, val_tfms=None, test_tfms=None):
    root = dataset_root(cfg)

    image_dir = root / cfg["image_subdir"]
    mask_dir  = root / cfg["mask_subdir"]
    splits    = root / cfg["splits_subdir"]

    train_ids = read_ids(splits / cfg["train_split"])
    val_txt_ids = read_ids(splits / cfg["val_split"])

    # split val.txt
    val_frac = cfg.get("val_fraction", 0.5)
    # val_ids, extra_train_ids = split_val_ids(val_txt_ids, val_frac)
    val_ids, test_ids = split_val_ids(val_txt_ids, val_frac)

    # final_train_ids = train_ids + extra_train_ids
    final_train_ids = train_ids 

    # dev mode shrink
    dbg = cfg.get("debug_samples")
    final_train_ids = maybe_truncate(final_train_ids, dbg)
    val_ids = maybe_truncate(val_ids, dbg)

    test_ids = maybe_truncate(test_ids, dbg)

    train_ds = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        ids=final_train_ids,
        transform=train_tfms,
    )

    val_ds = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        ids=val_ids,
        transform=val_tfms,
    )

    test_ds = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        ids=test_ids,
        transform=test_tfms,
    )

    return train_ds, val_ds, test_ds


# ---- dataloaders ----
def get_dataloaders(cfg_path="configs/default.yaml",
                    train_tfms=None,
                    val_tfms=None,
                    test_tfms=None):

    cfg = load_config(cfg_path)

    # train_ds, val_ds = get_datasets(cfg, train_tfms, val_tfms)
    test_tfms = val_tfms if test_tfms is None else test_tfms # if not provided, use val_tfms
    train_ds, val_ds, test_ds = get_datasets(cfg, train_tfms, val_tfms, test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle_train"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("test_batch_size", cfg["batch_size"]),
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    return train_loader, val_loader, test_loader




    



if __name__ == "__main__":
    print("src/data/loader.py executed")
    # cfg = load_config()
    # print("Config loaded:")
    # print(cfg)
    train_tfms = get_baseline_train_transforms(img_size=256)
    val_tfms   = get_baseline_val_transforms(img_size=256)
    test_tfms  = get_baseline_test_transforms(img_size=256)
    train_loader, val_loader, test_loader = get_dataloaders(train_tfms=train_tfms,
                                                            val_tfms=val_tfms,
                                                            test_tfms=test_tfms)
    print(f"Train loader: {len(train_loader.dataset)} samples")
    images, masks = next(iter(train_loader))
    print(f"Batch - images: {images.shape}, masks: {masks.shape}")

