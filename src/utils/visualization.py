from matplotlib.pylab import f
import numpy as np
import torch
import matplotlib.pyplot as plt


def to_numpy(t):
    """Move tensor to CPU & convert to numpy."""
    if torch.is_tensor(t):
        t = t.detach().cpu()
    return np.array(t)


def colorize_mask(mask, palette=None):
    """
    mask: (H, W) int array
    palette: list of RGB colors, length = num_classes
    """
    mask = to_numpy(mask)
    h, w = mask.shape

    if palette is None:
        # fallback palette
        rng = np.random.default_rng(0)
        palette = rng.integers(low=0, high=255, size=(mask.max()+1, 3))

    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        out[mask == cls] = color

    return out


def show_image_mask(image, mask=None, pred=None, palette=None, title=None, epoch=None):
    """
    image: (C, H, W) tensor
    mask, pred: (H, W) class indices
    """
    image = to_numpy(image)
    image = np.transpose(image, (1, 2, 0))  # CHW → HWC

    ncols = 1 + (mask is not None) + (pred is not None)
    fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    if ncols == 1:
        axs = [axs]

    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis("off")

    idx = 1

    if mask is not None:
        axs[idx].imshow(colorize_mask(mask, palette), alpha=0.9)
        axs[idx].set_title("Mask")
        axs[idx].axis("off")
        idx += 1

    if pred is not None:
        axs[idx].imshow(colorize_mask(pred, palette), alpha=0.9)
        axs[idx].set_title("Prediction")
        axs[idx].axis("off")
    
    title = f"Epoch {epoch}, {title}" if epoch is not None else title
    if title:
        fig.suptitle(title)

    plt.tight_layout()
    # plt.show()
    plt.show(block=False)
    plt.pause(0.001)



def visualize_batch(images, masks=None, max_items=8, epoch=None):
    """
    Show the first few batch samples.
    images: (B, C, H, W)
    masks:  (B, H, W) or None
    """
    b = min(images.shape[0], max_items)

    fig, axs = plt.subplots(b, 2 if masks is not None else 1,
                            figsize=(6*(2 if masks is not None else 1), 4*b))

    if b == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(b):
        img = to_numpy(images[i])
        img = np.transpose(img, (1, 2, 0))
        axs[i, 0].imshow(img)
        axs[i, 0].axis("off")

        if masks is not None:
            axs[i, 1].imshow(colorize_mask(masks[i]))
            axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
