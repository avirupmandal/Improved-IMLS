import os

import numpy as np
import torch
from PIL import Image


# def save_tensor_rgb_png(path: str, img_hwc: torch.Tensor) -> None:
#     """Save (H, W, 3) float RGB in [0, 1] as an 8-bit PNG."""
#     d = os.path.dirname(path)
#     if d:
#         os.makedirs(d, exist_ok=True)
#     arr = img_hwc.detach().cpu().clamp(0.0, 1.0).numpy()
#     arr = (arr * 255.0 + 0.5).astype(np.uint8)
#     Image.fromarray(arr).save(path)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
