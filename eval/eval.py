"""Unified evaluation utilities for DiffusionEdge.

This module provides one dataset-level entrypoint:
    evaluate_dataset(gt_dir, pred_dir, ...)

It computes:
    - generic pixelwise metrics (AP, ROC_AUC) via scikit-learn
    - optional BSDS-style edge metrics via pyEdgeEval when mode="edge"
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ALLOWED_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".npy", ".mat")


def _collect_numeric_arrays_from_mat(item: Any, target_field: str = "Boundaries") -> List[np.ndarray]:
    arrays: List[np.ndarray] = []

    if isinstance(item, np.void):
        field_names = item.dtype.names or ()
        if target_field in field_names:
            return _collect_numeric_arrays_from_mat(item[target_field], target_field=target_field)
        for field in field_names:
            arrays.extend(_collect_numeric_arrays_from_mat(item[field], target_field=target_field))
        return arrays

    if isinstance(item, np.ndarray):
        if item.dtype.names is not None:
            if target_field in item.dtype.names:
                return _collect_numeric_arrays_from_mat(item[target_field], target_field=target_field)
            for field in item.dtype.names:
                arrays.extend(_collect_numeric_arrays_from_mat(item[field], target_field=target_field))
            return arrays

        if item.dtype.kind == "O":
            for elem in item.flat:
                arrays.extend(_collect_numeric_arrays_from_mat(elem, target_field=target_field))
            return arrays

        if item.dtype.kind in "uifb":
            arrays.append(np.asarray(item, dtype=np.float32))
            return arrays

    return arrays


def _load_mat_gray01(path: str) -> np.ndarray:
    try:
        import scipy.io as scipy_io
    except Exception as exc:
        raise ImportError(f"Reading MATLAB files requires scipy: {path}") from exc

    mat = scipy_io.loadmat(path)
    keys = [key for key in mat.keys() if not key.startswith("__")]
    if not keys:
        raise ValueError(f"No valid keys found in MAT file: {path}")

    root = mat["groundTruth"] if "groundTruth" in mat else max((mat[k] for k in keys), key=lambda value: value.size)
    arrays = _collect_numeric_arrays_from_mat(root, target_field="Boundaries")
    if not arrays:
        for key in keys:
            arrays.extend(_collect_numeric_arrays_from_mat(mat[key], target_field="Boundaries"))

    processed: List[np.ndarray] = []
    for array in arrays:
        value = np.asarray(array, dtype=np.float32)
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        while value.ndim > 2 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim == 3 and value.shape[-1] == 3:
            value = 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]
        if value.ndim == 2:
            processed.append(value)

    if not processed:
        raise ValueError(f"Could not extract 2D arrays from MAT file: {path}")

    ref_shape = processed[0].shape
    same_shape = [arr for arr in processed if arr.shape == ref_shape]
    if not same_shape:
        raise ValueError(f"No shape-consistent arrays in MAT file: {path}")

    if len(same_shape) == 1:
        data = same_shape[0]
    else:
        data = np.mean(np.stack(same_shape, axis=0), axis=0)

    if not np.isfinite(data).all():
        raise ValueError(f"Found NaN/Inf in {path}")

    if float(data.max()) > 1.0 + 1e-6:
        data = data / 255.0
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def list_stems(dir_path: str) -> List[str]:
    """List filename stems under a directory for supported extensions."""
    if not osp.isdir(dir_path):
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")

    stems: List[str] = []
    for name in os.listdir(dir_path):
        stem, ext = osp.splitext(name)
        if ext.lower() in ALLOWED_EXTS:
            stems.append(stem)
    stems.sort()
    return stems


def pick_path(root: str, stem: str) -> str:
    """Return the first existing file for the given stem under root."""
    for ext in ALLOWED_EXTS:
        path = osp.join(root, stem + ext)
        if osp.isfile(path):
            return path
    raise FileNotFoundError(f"No supported file for stem '{stem}' under {root}")


def load_gray01(path: str) -> np.ndarray:
    """Load image/array as a single-channel float32 array in [0, 1]."""
    ext = osp.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".mat":
        arr = _load_mat_gray01(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            raise ValueError(f"Unexpected array rank/channels for {path}: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array for {path}, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Found NaN/Inf in {path}")

    arr_min = float(arr.min()) if arr.size else 0.0
    arr_max = float(arr.max()) if arr.size else 0.0

    if arr_min < -1e-6:
        raise ValueError(f"Negative values in {path}: min={arr_min:.6f}")

    if arr_max <= 1.0 + 1e-6:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    if arr_max <= 255.0 + 1e-6:
        return np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)

    raise ValueError(
        f"Unsupported value range in {path}: min={arr_min:.6f}, max={arr_max:.6f}. "
        "Expected [0,1] or [0,255]."
    )


def _normalize_thresholds(thresholds: Any) -> Any:
    """Normalize threshold input for pyEdgeEval (int/float/list)."""
    if isinstance(thresholds, (int, float)):
        return thresholds
    if isinstance(thresholds, str):
        stripped = thresholds.strip()
        if stripped.isdigit():
            return int(stripped)
        parts = [p for p in stripped.replace(" ", "").split(",") if p]
        if not parts:
            raise ValueError(
                f"Could not parse thresholds string '{thresholds}'. Use int or comma-separated floats."
            )
        return [float(x) for x in parts]
    if isinstance(thresholds, np.ndarray):
        return thresholds.tolist()
    if isinstance(thresholds, (list, tuple)):
        return [float(x) for x in thresholds]
    raise TypeError(
        f"Unsupported thresholds type: {type(thresholds)}. Expected int/float/list/tuple/ndarray/str."
    )


def _matched_stems(gt_dir: str, pred_dir: str) -> List[str]:
    gt_stems = set(list_stems(gt_dir))
    pred_stems = set(list_stems(pred_dir))
    stems = sorted(gt_stems & pred_stems)
    if not stems:
        raise FileNotFoundError(
            f"No matching stems between GT ({gt_dir}) and predictions ({pred_dir})."
        )
    return stems


def _compute_generic_metrics(gt_dir: str, pred_dir: str) -> Dict[str, Any]:
    """Compute pixelwise generic metrics."""
    stems = _matched_stems(gt_dir, pred_dir)

    y_true_list: List[np.ndarray] = []
    y_score_list: List[np.ndarray] = []

    for stem in stems:
        gt = load_gray01(pick_path(gt_dir, stem))
        pred = load_gray01(pick_path(pred_dir, stem))
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for '{stem}': GT={gt.shape}, pred={pred.shape}")

        y_true_list.append((gt > 0.5).astype(np.uint8).reshape(-1))
        y_score_list.append(pred.astype(np.float32).reshape(-1))

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

    return {
        "AP": float(average_precision_score(y_true, y_score)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
        "PR_precision": precision,
        "PR_recall": recall,
        "PR_thresholds": pr_thresholds,
        "ROC_FPR": fpr,
        "ROC_TPR": tpr,
        "ROC_thresholds": roc_thresholds,
        "N": int(y_true.size),
    }


def _edge_eval_single(kwargs: Dict[str, Any]):
    """Top-level pyEdgeEval worker callback (must be picklable)."""
    from pyEdgeEval.common.binary_label.evaluate_boundaries import evaluate_boundaries_threshold
    from pyEdgeEval.common.utils import check_thresholds

    eval_thresholds = check_thresholds(kwargs["thresholds"])
    gt = load_gray01(kwargs["gt_path"])
    pred = load_gray01(kwargs["pred_path"])
    gt_bin = (gt > 0.5).astype(np.uint8)
    return evaluate_boundaries_threshold(
        thresholds=eval_thresholds,
        pred=pred,
        gt=gt_bin,
        max_dist=kwargs["max_dist"],
        apply_thinning=kwargs["apply_thinning"],
        apply_nms=kwargs["apply_nms"],
    )


def _compute_edge_metrics(
    gt_dir: str,
    pred_dir: str,
    thresholds: Any,
    max_dist: float,
    apply_thinning: bool,
    apply_nms: bool,
    nproc: int,
    save_dir: Optional[str],
) -> Dict[str, Any]:
    """Compute edge metrics using pyEdgeEval."""
    try:
        from pyEdgeEval.common.binary_label import calculate_metrics, save_results
    except Exception as exc:
        raise ImportError(
            "mode='edge' requires pyEdgeEval. Install it or use mode='binary'."
        ) from exc

    stems = _matched_stems(gt_dir, pred_dir)
    thresholds = _normalize_thresholds(thresholds)

    samples: List[Dict[str, Any]] = []
    for stem in stems:
        samples.append(
            {
                "name": stem,
                "thresholds": thresholds,
                "gt_path": pick_path(gt_dir, stem),
                "pred_path": pick_path(pred_dir, stem),
                "max_dist": max_dist,
                "apply_thinning": apply_thinning,
                "apply_nms": apply_nms,
            }
        )

    sample_metrics, threshold_metrics, overall_metrics = calculate_metrics(
        eval_single=_edge_eval_single,
        thresholds=thresholds,
        samples=samples,
        nproc=nproc,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_results(
            root=save_dir,
            sample_metrics=sample_metrics,
            threshold_metrics=threshold_metrics,
            overall_metric=overall_metrics,
        )

    return overall_metrics


def flatten_metrics(result: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten evaluation output into scalar-only dictionary for logging."""
    out: Dict[str, float] = {}

    generic = result.get("generic_metrics", {})
    for key in ("AP", "ROC_AUC", "N"):
        value = generic.get(key)
        if value is not None and np.isscalar(value):
            out[f"{prefix}generic/{key}"] = float(value)

    edge = result.get("edge_metrics", None)
    if edge is not None:
        for key, value in edge.items():
            if np.isscalar(value):
                out[f"{prefix}edge/{key}"] = float(value)

    return out


def evaluate_dataset(
    gt_dir: str,
    pred_dir: str,
    mode: str = "binary",
    thresholds: Any = 99,
    max_dist: float = 0.0075,
    apply_thinning: bool = False,
    apply_nms: bool = False,
    nproc: int = 4,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate paired predictions and ground truth maps.

    Args:
        gt_dir: directory containing GT maps.
        pred_dir: directory containing prediction maps.
        mode: "binary" for generic metrics only, or "edge" for pyEdgeEval + generic.
        thresholds: threshold setting for edge mode.
        max_dist: pyEdgeEval matching tolerance.
        apply_thinning: pyEdgeEval thinning.
        apply_nms: pyEdgeEval NMS.
        nproc: pyEdgeEval workers.
        save_dir: optional path for edge-mode JSON outputs.
    """
    if mode not in {"binary", "edge"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'binary' or 'edge'.")
    if not osp.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not osp.isdir(pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    generic_metrics = _compute_generic_metrics(gt_dir, pred_dir)
    edge_metrics: Optional[Dict[str, Any]] = None

    if mode == "edge":
        edge_metrics = _compute_edge_metrics(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            thresholds=thresholds,
            max_dist=max_dist,
            apply_thinning=apply_thinning,
            apply_nms=apply_nms,
            nproc=nproc,
            save_dir=save_dir,
        )

    return {
        "edge_metrics": edge_metrics,
        "generic_metrics": generic_metrics,
    }
