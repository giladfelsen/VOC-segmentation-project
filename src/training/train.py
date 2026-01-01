import shutil
import argparse
from logging import config
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.data.loader import get_dataloaders
from src.data.transforms import (
    get_baseline_train_transforms,
    get_baseline_val_transforms,
)

import src.data.transforms as transforms_module
from src.models.model_factory import build_model
from src.metrics.segmentation import compute_segmentation_metrics
from src.optim.factory import build_optimizer
from src.scheduler.factory import build_scheduler
from src.losses.factory import build_loss_function

from tqdm.auto import tqdm

import random
from datetime import datetime # Will be used to determine which images to delete

from src.utils.visualization import show_image_mask, to_numpy, colorize_mask
import matplotlib.pyplot as plt

from src.losses.segmentation import cross_entropy_loss




# ------------------------------
# Utils
# ------------------------------
def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config yaml",
    )
    return parser.parse_args()

#
#
#
@torch.no_grad()
def visualize_and_save_random_batch(
    model,
    loader,
    device,
    num_classes,
    save_dir="debug_vis",
    max_items=6,
    max_files=200,
    epoch=None,
    run_name=None,
):
    """
    Saves image / mask / prediction triplets to disk from a random batch.
    Skips visualization entirely if save_dir already has > max_files.
    """

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # ---- storage cap check ----
    existing_files = list(save_path.glob("*.png"))
    if len(existing_files) >= max_files:
        print(f"[VIS] Skipping visualization (>{max_files} files already).")
        return

    # ---- pick random batch ----
    model.eval()
    idx = random.randint(0, len(loader) - 1)

    for i, (images, masks) in enumerate(loader):
        if i != idx:
            continue

        images = images.to(device)
        masks = masks.long().to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for k in range(min(images.shape[0], max_items)):

            img = to_numpy(images[k]).transpose(1, 2, 0)
            gt_mask = colorize_mask(masks[k])
            pred_mask = colorize_mask(preds[k])

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(img)
            axs[0].set_title("Image")
            axs[0].axis("off")

            axs[1].imshow(gt_mask)
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            axs[2].imshow(pred_mask)
            axs[2].set_title("Prediction")
            axs[2].axis("off")

            file_name = f"{run_name}_{epoch}"
            fname = save_path / f"{file_name}_{timestamp}_sample{k}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        break

# ------------------------------
# Training + Evaluation
# ------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, ignore_index=None):
    model = model.to(device)
    model.train()
    total_loss = 0.0

    

    progress = tqdm(loader, desc="Train", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = loss_fn(logits, masks, ignore_index=ignore_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        
        
        # live update bar
        progress.set_postfix(loss=f"{loss.item():.4f}")

        

    return total_loss / len(loader)



@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes, ignore_index=None):
    eval_device = "cpu" if device.type == "mps" else device
    model = model.to(eval_device)
    model.eval()

    total_loss = 0.0
    metrics_buffer = []

    progress = tqdm(loader, desc="Val", leave=False)

    for images, masks in progress:
        images = images.to(eval_device)
        masks = masks.to(eval_device)

        logits = model(images)

        loss = loss_fn(
            logits,
            masks,
            ignore_index=ignore_index,
        )
        total_loss += loss.item()

        batch_metrics = compute_segmentation_metrics(
            logits,
            masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        metrics_buffer.append(batch_metrics)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_metrics = {
        k: sum(m[k] for m in metrics_buffer) / len(metrics_buffer)
        for k in metrics_buffer[0].keys()
        if k != "per_class_iou"
    }
    model = model.to(device) # put device back on original device

    return {
        "loss": total_loss / len(loader),
        **avg_metrics,
    }



# ------------------------------
# Main entrypoint
# ------------------------------


def main():
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    saving_checkpoint_name = f"checkpoints/run_{time_stamp}"
    Path(saving_checkpoint_name).mkdir(parents=True, exist_ok=False)

    args = parse_args()
    cfg = load_config(args.config)

    print(f"Using config: {args.config}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    shutil.copy(args.config, f"{saving_checkpoint_name}/config.yaml") # Save config file used for this run with checkpoints.


    num_classes = cfg["model"]["num_classes"]
    ignore_index = cfg.get("metrics", {}).get("ignore_index", None)

    # --- dataloaders ---
    train_trans_type = cfg.get("transform_pipeline", "default")
    if train_trans_type == "basic":
        train_tfms = transforms_module.get_basic_train_transforms(cfg["image_size"])
    else:
        train_tfms = get_baseline_train_transforms(cfg["image_size"])

    val_tfms = get_baseline_val_transforms(cfg["image_size"])

    train_loader, val_loader, _ = get_dataloaders(
        cfg_path=args.config,
        train_tfms=train_tfms,
        val_tfms=val_tfms,
    )

    # --- model ---
    model_checkpoint = cfg.get("model_checkpoint", None)
    if model_checkpoint is not None:
        print(f"Loading model checkpoint from: {model_checkpoint}")
        model = build_model(**cfg["model"])
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model = model.to(device)
    else:
        model = build_model(**cfg["model"]).to(device)

    # --- optimizer ---
    optimizer = build_optimizer(model, cfg)


    # --- scheduler#= ---
    # TODO
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # --- loss ---
    loss_fn = build_loss_function(cfg)

    # --- logging ---
    # writer = SummaryWriter(log_dir="runs/baseline")
    writer = SummaryWriter(log_dir=f"runs/{time_stamp}")


    best_val_loss = float("inf")
    
    best_metrics = {"miou": 0.0, "dice": 0.0, "pixel_acc": 0.0}

    for epoch in range(cfg["trainer"]["epochs"]):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            ignore_index=ignore_index,
        )

        val_stats = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        print(
            f"Epoch {epoch+1}/{cfg['trainer']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"mIoU={val_stats['miou']:.4f} | "
            f"Dice={val_stats['dice']:.4f} | "
            f"PixelAcc={val_stats['pixel_acc']:.4f}"
        )

        # tensorboard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_stats["loss"], epoch)
        writer.add_scalar("metrics/mIoU", val_stats["miou"], epoch)
        writer.add_scalar("metrics/Dice", val_stats["dice"], epoch)
        writer.add_scalar("metrics/PixelAcc", val_stats["pixel_acc"], epoch)

        # checkpoint best model
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            Path(saving_checkpoint_name).mkdir(exist_ok=True)
            # torch.save(model.state_dict(), f"{saving_checkpoint_name}/best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_stats["loss"],
            }, f"{saving_checkpoint_name}/best_model.pt")


        for key in best_metrics.keys():
            val = best_metrics[key]
            if val_stats[key] > val:
                best_metrics[key] = val_stats[key]
                print(f"New best {key}: {val_stats[key]:.4f}")
                # torch.save(model.state_dict(), f"{saving_checkpoint_name}/best_model_{key}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_stats["loss"],
                }, f"{saving_checkpoint_name}/best_{key}_model.pt")

        # visualize model predictions on val set
        if cfg.get("visualize_model_preds", False):
            n = cfg.get("visualize_every", 1)
            if (epoch + 1) % n == 0:
                config_name = cfg.get("config_name", None)
                visualize_and_save_random_batch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    num_classes=num_classes,
                    max_items=3,
                    max_files=200,
                    epoch=epoch+1,
                    run_name=config_name
                )

    writer.close()


if __name__ == "__main__":
    main()
