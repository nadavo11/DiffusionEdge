"""Preview generation utilities for DiffusionEdge evaluation."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .eval import load_gray01, pick_path


def _to_rgb01(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def _to_uint8_rgb(arr_01: np.ndarray) -> np.ndarray:
    if arr_01.ndim == 2:
        arr_01 = np.repeat(arr_01[..., None], 3, axis=2)
    arr_01 = np.clip(arr_01, 0.0, 1.0)
    return (arr_01 * 255.0 + 0.5).astype(np.uint8)


def _colorize(mask_01: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    mask_01 = np.clip(mask_01.astype(np.float32), 0.0, 1.0)
    mask_u8 = (mask_01 * 255.0 + 0.5).astype(np.uint8)
    color_bgr = cv2.applyColorMap(mask_u8, cmap)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def make_previews(
    img_dir: str,
    pred_dir: str,
    gt_dir: str,
    out_dir: str,
    stems: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    img_exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm"),
    map_exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".npy"),
    suffix: str = "_preview",
) -> List[str]:
    """Create [Input | GT | Pred | Overlay] preview panels."""
    os.makedirs(out_dir, exist_ok=True)

    if stems is None:
        pred_stems = sorted({os.path.splitext(name)[0] for name in os.listdir(pred_dir)})
        gt_stems = {os.path.splitext(name)[0] for name in os.listdir(gt_dir)}
        stems = [stem for stem in pred_stems if stem in gt_stems]

    stems = list(stems)
    if limit is not None:
        stems = stems[: max(0, int(limit))]

    written: List[str] = []
    for stem in stems:
        try:
            img_path = None
            pred_path = None
            gt_path = None

            for ext in img_exts:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break

            for ext in map_exts:
                candidate = os.path.join(pred_dir, stem + ext)
                if os.path.isfile(candidate):
                    pred_path = candidate
                    break

            for ext in map_exts:
                candidate = os.path.join(gt_dir, stem + ext)
                if os.path.isfile(candidate):
                    gt_path = candidate
                    break

            if img_path is None or pred_path is None or gt_path is None:
                continue

            rgb = _to_rgb01(img_path)
            pred = load_gray01(pred_path)
            gt = load_gray01(gt_path)

            h, w = pred.shape
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            if gt.shape != (h, w):
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

            pred_rgb = _colorize(pred)
            gt_rgb = _colorize(gt)
            overlay = np.clip(0.5 * rgb + 0.5 * pred_rgb, 0.0, 1.0)

            panel = np.concatenate(
                [
                    _to_uint8_rgb(rgb),
                    _to_uint8_rgb(gt_rgb),
                    _to_uint8_rgb(pred_rgb),
                    _to_uint8_rgb(overlay),
                ],
                axis=1,
            )

            out_path = os.path.join(out_dir, f"{stem}{suffix}.png")
            ok = cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            if ok:
                written.append(out_path)
        except Exception:
            continue

    return written

