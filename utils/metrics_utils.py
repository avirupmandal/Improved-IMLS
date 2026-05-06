# import json
# import os
# import re
# from typing import Dict, List, Optional

# import torch

# from fused_ssim import fused_ssim
# from utils.image_utils import psnr, save_tensor_rgb_png


# def _lpips_module():
#     from lpipsPyTorch.modules.lpips import LPIPS

#     return LPIPS


# def make_lpips(net_type: str = "alex", device: Optional[torch.device] = None):
#     LPIPS = _lpips_module()
#     dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     m = LPIPS(net_type=net_type, version="0.1").to(dev)
#     m.eval()
#     for p in m.parameters():
#         p.requires_grad_(False)
#     return m


# @torch.no_grad()
# def compute_view_metrics(
#     pred_hwc: torch.Tensor,
#     gt_hwc: torch.Tensor,
#     lpips_model: torch.nn.Module,
# ) -> Dict[str, float]:
#     """
#     pred_hwc, gt_hwc: (H, W, 3) float on CUDA, roughly [0, 1] RGB (same as training).
#     """
#     pred_hwc = pred_hwc.clamp(0.0, 1.0)
#     gt_hwc = gt_hwc.clamp(0.0, 1.0)

#     mse = ((pred_hwc - gt_hwc) ** 2).mean()

#     pred_n = pred_hwc.permute(2, 0, 1).unsqueeze(0)
#     gt_n = gt_hwc.permute(2, 0, 1).unsqueeze(0)
#     psnr_v = psnr(pred_n, gt_n).mean()
#     ssim_v = fused_ssim(pred_n, gt_n, train=False)
#     lpips_v = lpips_model(pred_n, gt_n).mean()

#     return {
#         "mse": float(mse.item()),
#         "psnr": float(psnr_v.item()),
#         "ssim": float(ssim_v.item()),
#         "lpips": float(lpips_v.item()),
#     }


# @torch.no_grad()
# def accumulate_metrics(
#     imlsplat,
#     gaussians,
#     cameras: list,
#     bg: torch.Tensor,
#     lpips_model: torch.nn.Module,
# ) -> Dict[str, float]:
#     if len(cameras) == 0:
#         return {}

#     totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
#     n = 0
#     for viewpoint_cam in cameras:
#         meshdict = imlsplat(gaussians, viewpoint_cam, bg)
#         pred = meshdict["image"][..., 0:3]

#         gt = viewpoint_cam.original_image.cuda().permute(1, 2, 0)
#         if viewpoint_cam.gt_alpha_mask is not None:
#             m = viewpoint_cam.gt_alpha_mask.cuda().permute(1, 2, 0)
#             gt = gt * m + bg * (1.0 - m)

#         mets = compute_view_metrics(pred, gt, lpips_model)
#         for k in totals:
#             totals[k] += mets[k]
#         n += 1

#     return {k: totals[k] / n for k in totals}


# @torch.no_grad()
# def save_test_view_renders(
#     imlsplat,
#     gaussians,
#     cameras: list,
#     bg: torch.Tensor,
#     out_dir: str,
# ) -> None:
#     """Render each test camera with scene background `bg` and write PNGs under out_dir."""
#     if len(cameras) == 0:
#         return
#     os.makedirs(out_dir, exist_ok=True)
#     for idx, cam in enumerate(cameras):
#         meshdict = imlsplat(gaussians, cam, bg)
#         pred = meshdict["image"][..., 0:3]
#         base = cam.image_name
#         base = os.path.basename(base.replace("\\", "/"))
#         stem, _ = os.path.splitext(base)
#         stem = re.sub(r'[<>:"/\\|?*]+', "_", stem) or "view"
#         path = os.path.join(out_dir, "{:04d}_{}.png".format(idx, stem))
#         save_tensor_rgb_png(path, pred)


# def _metrics_filename(exp_name: str) -> str:
#     name = (exp_name or "").strip()
#     if not name:
#         return "metrics.json"
#     safe = re.sub(r'[<>:"/\\|?*]+', "_", name).strip("._")
#     if not safe:
#         safe = "run"
#     return f"metrics_{safe}.json"


# def append_metrics_json(
#     model_path: str, iteration: int, metrics: Dict[str, float], exp_name: str = ""
# ) -> None:
#     path = os.path.join(model_path, _metrics_filename(exp_name))
#     row = {"iteration": iteration, **metrics}
#     history: List[Dict] = []
#     if os.path.isfile(path):
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 loaded = json.load(f)
#             if isinstance(loaded, list):
#                 history = loaded
#         except (json.JSONDecodeError, OSError):
#             history = []
#     history.append(row)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2)
