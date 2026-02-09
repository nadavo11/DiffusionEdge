import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.data import EdgeDataset, EdgeDatasetTest
from denoising_diffusion_pytorch.ema import EMA
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.utils import create_logger, cycle, dict2str, exists, has_int_squareroot
from eval.eval import evaluate_dataset, flatten_metrics
from eval.previews import make_previews
from eval.run_params import save_run_params

try:
    import wandb as _wandb

    if not hasattr(_wandb, "init"):
        raise ImportError("Imported module 'wandb' does not expose 'init' (likely namespace shadowing).")
    wandb = _wandb
    WANDB_ENABLED = True
except Exception as wandb_error:
    WANDB_ENABLED = False

    class _WandbStub:
        class Image:
            def __init__(self, *args, **kwargs):
                pass

        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _WandbStub()
    print(f"[WARN] WandB disabled: {wandb_error}")


_GT_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".npy", ".mat")


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


def _load_mat_edge01(path: Path) -> np.ndarray:
    try:
        import scipy.io as scipy_io
    except Exception as exc:
        raise ImportError(f"Reading MATLAB GT requires scipy for file: {path}") from exc

    mat = scipy_io.loadmat(str(path))
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
        raise ValueError(f"Could not extract 2D edge arrays from MAT file: {path}")

    ref_shape = processed[0].shape
    same_shape = [arr for arr in processed if arr.shape == ref_shape]
    if not same_shape:
        raise ValueError(f"No shape-consistent edge arrays in MAT file: {path}")

    if len(same_shape) == 1:
        data = same_shape[0]
    else:
        data = np.mean(np.stack(same_shape, axis=0), axis=0)

    if not np.isfinite(data).all():
        raise ValueError(f"MAT GT contains NaN/Inf: {path}")

    if data.max() > 1.0:
        data = data / 255.0
    data = np.clip(data, 0.0, 1.0)
    return data.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional latent diffusion model")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file: str, conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if conf is None:
        conf = {}
    with open(config_file, "r", encoding="utf-8") as handle:
        exp_conf = yaml.load(handle, Loader=yaml.FullLoader)
        for key, value in exp_conf.items():
            conf[key] = value
    return conf


def _cfg_to_dict(value: Any) -> Any:
    if isinstance(value, CfgNode):
        return {k: _cfg_to_dict(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _cfg_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cfg_to_dict(v) for v in value]
    return value


def _save_gray01_png(path: Path, arr_01: torch.Tensor) -> None:
    tensor = arr_01.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor[0]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    image_u8 = (tensor * 255.0 + 0.5).to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_u8)
    if not ok:
        raise IOError(f"Failed to write PNG: {path}")


def _load_gt01(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(str(path)).astype(np.float32)
    elif suffix == ".mat":
        data = _load_mat_edge01(path)
    else:
        data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise FileNotFoundError(f"Cannot read GT file: {path}")
        data = data.astype(np.float32)

    if data.ndim == 3:
        data = data[..., 0]
    if data.ndim != 2:
        raise ValueError(f"GT must be 2D, got shape={data.shape} for {path}")

    if not np.isfinite(data).all():
        raise ValueError(f"GT contains NaN/Inf: {path}")

    if data.max() > 1.0:
        data = data / 255.0
    data = np.clip(data, 0.0, 1.0)
    data = (data > 0.5).astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0)


def _to_pred01(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().cpu().float()
    if out.dim() == 2:
        out = out.unsqueeze(0)
    elif out.dim() == 4:
        if out.shape[1] > 1:
            out = out[:, :1]
    elif out.dim() == 3:
        if out.shape[0] > 1:
            out = out[:1]
    else:
        raise ValueError(f"Prediction tensor must be 2D/3D/4D, got {tuple(out.shape)}")

    if float(out.min()) < 0.0 or float(out.max()) > 1.0:
        out = (out + 1.0) * 0.5

    return torch.clamp(out, 0.0, 1.0)


def _tensor_to_wandb_image(tensor: torch.Tensor, denorm: bool = False) -> np.ndarray:
    value = tensor.detach().cpu().float()
    if value.dim() == 4:
        value = value[0]
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() != 3:
        raise ValueError(f"Expected tensor with 3 dims (C,H,W), got {tuple(value.shape)}")

    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)

    if denorm:
        value = (value + 1.0) * 0.5

    value = torch.clamp(value, 0.0, 1.0)
    return (value.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _resolve_eval_dirs(
    data_root: Path,
    eval_split: str,
    image_dir_override: Optional[str],
    gt_dir_override: Optional[str],
) -> Tuple[Path, Path]:
    if image_dir_override:
        image_dir = Path(image_dir_override)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Eval image_dir does not exist: {image_dir}")
    else:
        image_candidates = [
            data_root / "image" / eval_split,
            data_root / "images" / eval_split,
            data_root / "image",
            data_root / "images",
        ]
        image_dir = next((path for path in image_candidates if path.is_dir()), None)
        if image_dir is None:
            raise FileNotFoundError(
                f"Could not resolve eval image directory under {data_root}. "
                f"Expected image/<split>, images/<split>, image, or images."
            )

    if gt_dir_override:
        gt_dir = Path(gt_dir_override)
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Eval gt_dir does not exist: {gt_dir}")
    else:
        gt_candidates = [
            data_root / "edge" / eval_split,
            data_root / "edges" / eval_split,
            data_root / "edge",
            data_root / "edges",
        ]
        gt_dir = next((path for path in gt_candidates if path.is_dir()), None)
        if gt_dir is None:
            raise FileNotFoundError(
                f"Could not resolve eval GT directory under {data_root}. "
                f"Expected edge/<split>, edges/<split>, edge, or edges."
            )

    return image_dir, gt_dir


def _build_eval_target(
    name: str,
    target_cfg: Dict[str, Any],
    default_data_root: Path,
    default_split: str,
    default_workers: int,
    image_size: Sequence[int],
) -> Dict[str, Any]:
    data_root = Path(target_cfg.get("data_root", default_data_root))
    eval_split = str(target_cfg.get("split", default_split))
    eval_batch_size = int(target_cfg.get("batch_size", 1))
    eval_workers = int(target_cfg.get("num_workers", default_workers))
    eval_image_override = target_cfg.get("image_dir")
    eval_gt_override = target_cfg.get("gt_dir")

    eval_image_dir, eval_gt_dir = _resolve_eval_dirs(
        data_root=data_root,
        eval_split=eval_split,
        image_dir_override=eval_image_override,
        gt_dir_override=eval_gt_override,
    )

    eval_dataset = EdgeDatasetTest(
        data_root=str(eval_image_dir),
        image_size=image_size,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=eval_workers,
    )

    return {
        "name": name,
        "log_prefix": str(target_cfg.get("log_prefix", name)),
        "loader": eval_loader,
        "image_dir": eval_image_dir,
        "gt_dir": eval_gt_dir,
        "num_batches": int(target_cfg.get("num_batches", -1)),
        "mode": str(target_cfg.get("mode", "edge")),
        "thresholds": target_cfg.get("thresholds", 99),
        "max_dist": float(target_cfg.get("max_dist", 0.0075)),
        "apply_nms": bool(target_cfg.get("apply_nms", False)),
        "apply_thinning": bool(target_cfg.get("apply_thinning", False)),
        "nproc": int(target_cfg.get("nproc", 4)),
        "preview_limit": int(target_cfg.get("preview_limit", 12)),
        "select_best_metric": str(target_cfg.get("select_best_metric", "generic/AP")),
        "dataset_size": len(eval_dataset),
    }


def main(args):
    cfg = CfgNode(args.cfg)

    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    if model_cfg.model_name == "cond_unet":
        from denoising_diffusion_pytorch.mask_cond_unet import Unet

        unet_cfg = model_cfg.unet
        unet = Unet(
            dim=unet_cfg.dim,
            channels=unet_cfg.channels,
            dim_mults=unet_cfg.dim_mults,
            learned_variance=unet_cfg.get("learned_variance", False),
            out_mul=unet_cfg.out_mul,
            cond_in_dim=unet_cfg.cond_in_dim,
            cond_dim=unet_cfg.cond_dim,
            cond_dim_mults=unet_cfg.cond_dim_mults,
            window_sizes1=unet_cfg.window_sizes1,
            window_sizes2=unet_cfg.window_sizes2,
            fourier_scale=unet_cfg.fourier_scale,
            cfg=unet_cfg,
        )
    else:
        raise NotImplementedError

    if model_cfg.model_type == "const_sde":
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f"{model_cfg.model_type} is not surportted !")

    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get("default_scale", False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get("use_l1", True),
        cfg=model_cfg,
    )

    data_cfg = cfg.data
    if data_cfg["name"] != "edge":
        raise NotImplementedError(f"Unsupported dataset type: {data_cfg['name']}")

    train_split = data_cfg.get("train_split", "train")
    dataset = EdgeDataset(
        data_root=data_cfg.img_folder,
        image_size=model_cfg.image_size,
        augment_horizontal_flip=data_cfg.augment_horizontal_flip,
        split=train_split,
        cfg=data_cfg,
    )
    dl = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=data_cfg.get("num_workers", 2),
    )

    train_cfg = cfg.trainer
    eval_cfg = train_cfg.get("eval", {})
    eval_cfg = _cfg_to_dict(eval_cfg)

    eval_enabled = bool(eval_cfg.get("enabled", train_cfg.get("test_before", True)))
    eval_targets: List[Dict[str, Any]] = []

    if eval_enabled:
        primary_cfg = dict(eval_cfg)
        rwtd_cfg = _cfg_to_dict(primary_cfg.pop("rwtd", {}))

        primary_target = _build_eval_target(
            name="val",
            target_cfg=primary_cfg,
            default_data_root=Path(data_cfg.img_folder),
            default_split=str(primary_cfg.get("split", data_cfg.get("eval_split", "val"))),
            default_workers=int(data_cfg.get("num_workers", 2)),
            image_size=model_cfg.image_size,
        )
        eval_targets.append(primary_target)
        print(
            f"[Eval:{primary_target['name']}] "
            f"image_dir={primary_target['image_dir']}, gt_dir={primary_target['gt_dir']}, "
            f"samples={primary_target['dataset_size']}, num_batches={primary_target['num_batches']}"
        )

        if bool(rwtd_cfg.get("enabled", False)):
            rwtd_target = _build_eval_target(
                name="rwtd",
                target_cfg=rwtd_cfg,
                default_data_root=Path(rwtd_cfg.get("data_root", data_cfg.img_folder)),
                default_split=str(rwtd_cfg.get("split", "val")),
                default_workers=int(data_cfg.get("num_workers", 2)),
                image_size=model_cfg.image_size,
            )
            eval_targets.append(rwtd_target)
            print(
                f"[Eval:{rwtd_target['name']}] "
                f"image_dir={rwtd_target['image_dir']}, gt_dir={rwtd_target['gt_dir']}, "
                f"samples={rwtd_target['dataset_size']}, num_batches={rwtd_target['num_batches']}"
            )
    else:
        print("[Eval] Disabled by configuration")

    trainer = Trainer(
        ldm,
        dl,
        train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr,
        train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every,
        results_folder=train_cfg.results_folder,
        amp=train_cfg.amp,
        fp16=train_cfg.fp16,
        log_freq=train_cfg.log_freq,
        cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get("weight_decay", 1e-4),
        eval_targets=eval_targets,
        eval_cfg=eval_cfg,
    )

    trainer.train()


class Trainer(object):
    def __init__(
        self,
        model,
        data_loader,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_wd=1e-4,
        train_num_steps=100000,
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        log_freq=20,
        resume_milestone=0,
        cfg={},
        eval_targets: Optional[List[Dict[str, Any]]] = None,
        eval_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision="fp16" if fp16 else "no",
            kwargs_handlers=[ddp_handler],
        )
        self.enable_resume = cfg.trainer.get("enable_resume", False)
        self.accelerator.native_amp = amp

        self.model = model

        assert has_int_squareroot(num_samples), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model.image_size

        self.best_loss = float("inf")
        self.eval_cfg = eval_cfg or {}
        self.eval_targets = eval_targets or []
        self.eval_enabled = bool(self.eval_targets)
        self.eval_every_steps = int(self.eval_cfg.get("every_steps", self.save_and_sample_every))
        self.eval_at_step0 = bool(self.eval_cfg.get("run_at_step0", cfg.trainer.get("test_before", True)))
        self.best_eval_scores: Dict[str, float] = {}
        self.best_eval_steps: Dict[str, int] = {}

        if self.eval_enabled:
            if self.eval_every_steps <= 0:
                raise ValueError("trainer.eval.every_steps must be > 0 when eval is enabled")
            for target in self.eval_targets:
                target_name = str(target.get("name", "unknown"))
                target_loader = target.get("loader")
                target_image_dir = target.get("image_dir")
                target_gt_dir = target.get("gt_dir")
                if target_loader is None:
                    raise ValueError(f"Eval target '{target_name}' has no loader")
                if target_image_dir is None or not Path(target_image_dir).is_dir():
                    raise FileNotFoundError(f"Eval target '{target_name}' image_dir not found: {target_image_dir}")
                if target_gt_dir is None or not Path(target_gt_dir).is_dir():
                    raise FileNotFoundError(f"Eval target '{target_name}' gt_dir not found: {target_gt_dir}")

                self.best_eval_scores[target_name] = float("-inf")
                self.best_eval_steps[target_name] = -1

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        self.opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_lr,
            weight_decay=train_wd,
        )
        lr_lambda = lambda iteration: max((1 - iteration / train_num_steps) ** 0.96, cfg.trainer.min_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.eval_output_root = self.results_folder / "eval"
            self.eval_output_root.mkdir(exist_ok=True, parents=True)

            self.ema = EMA(
                model,
                ema_model=None,
                beta=0.999,
                update_after_step=cfg.trainer.ema_update_after_step,
                update_every=cfg.trainer.ema_update_every,
            )

            wandb.init(
                project=cfg.get("project_name", "diffusion-model"),
                name=cfg.get("run_name", None),
                config=_cfg_to_dict(cfg),
                resume="allow" if self.enable_resume else False,
            )

            save_run_params(_cfg_to_dict(cfg), self.results_folder, prefix="train")

        self.step = 0

        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.opt,
            self.lr_scheduler,
        )
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)

        resume_file = str(self.results_folder / f"model-{resume_milestone}.pt")
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        filename = f"model-{milestone}.pt"
        save_path = str(self.results_folder / filename)

        if self.enable_resume:
            data = {
                "step": self.step,
                "best_loss": self.best_loss,
                "best_eval_scores": self.best_eval_scores,
                "best_eval_steps": self.best_eval_steps,
                "model": self.accelerator.get_state_dict(self.model),
                "opt": self.opt.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "ema": self.ema.state_dict(),
                "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            }
            torch.save(data, save_path)
        else:
            data = {
                "model": self.accelerator.get_state_dict(self.model),
                "best_loss": self.best_loss,
                "best_eval_scores": self.best_eval_scores,
                "best_eval_steps": self.best_eval_steps,
            }
            torch.save(data, save_path)

    def load(self, milestone):
        assert self.enable_resume, "resume is available only if self.enable_resume is True !"

        filename = f"model-{milestone}.pt"
        load_path = str(self.results_folder / filename)
        data = torch.load(load_path, map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])
        if "scale_factor" in data["model"]:
            model.scale_factor = data["model"]["scale_factor"]

        self.step = int(data["step"])
        self.opt.load_state_dict(data["opt"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

        self.best_loss = float(data.get("best_loss", float("inf")))
        if "best_eval_scores" in data and isinstance(data["best_eval_scores"], dict):
            loaded_scores = {str(k): float(v) for k, v in data["best_eval_scores"].items()}
        else:
            # Backward compatibility with older checkpoints.
            loaded_scores = {"val": float(data.get("best_eval_score", float("-inf")))}
        if "best_eval_steps" in data and isinstance(data["best_eval_steps"], dict):
            loaded_steps = {str(k): int(v) for k, v in data["best_eval_steps"].items()}
        else:
            loaded_steps = {"val": int(data.get("best_eval_step", -1))}

        for target in self.eval_targets:
            target_name = str(target.get("name", "val"))
            self.best_eval_scores[target_name] = float(loaded_scores.get(target_name, float("-inf")))
            self.best_eval_steps[target_name] = int(loaded_steps.get(target_name, -1))

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

        print(
            f"Resumed from {load_path} at step {self.step}. "
            f"Best loss={self.best_loss:.6f}, best eval={self.best_eval_scores}"
        )

    @staticmethod
    def _extract_raw_sizes(raw_size: Any, batch_size: int) -> List[Tuple[int, int]]:
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            width_like, height_like = raw_size
            if torch.is_tensor(width_like) and torch.is_tensor(height_like):
                widths = width_like.detach().cpu().tolist()
                heights = height_like.detach().cpu().tolist()
                return [(int(w), int(h)) for w, h in zip(widths, heights)]

        if torch.is_tensor(raw_size):
            values = raw_size.detach().cpu().numpy()
            if values.ndim == 2 and values.shape[1] == 2:
                return [(int(item[0]), int(item[1])) for item in values]

        if isinstance(raw_size, (list, tuple)) and len(raw_size) == batch_size:
            sizes: List[Tuple[int, int]] = []
            for item in raw_size:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    sizes.append((int(item[0]), int(item[1])))
                else:
                    sizes.append((-1, -1))
            return sizes

        return [(-1, -1)] * batch_size

    @staticmethod
    def _find_gt_path_in_dir(gt_dir: Path, stem: str) -> Optional[Path]:
        for ext in _GT_EXTS:
            candidate = gt_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def _run_model_sample(self, cond: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            output = self.model.module.sample(batch_size=cond.shape[0], cond=cond, mask=mask)
        elif isinstance(self.model, nn.Module):
            output = self.model.sample(batch_size=cond.shape[0], cond=cond, mask=mask)
        else:
            raise NotImplementedError("Unexpected model wrapper type")

        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError(f"Model.sample returned non-tensor output of type {type(output)}")
        return output

    def _save_batch_sample(self, batch: Dict[str, Any], milestone: int, step: int) -> None:
        self.model.eval()
        with torch.no_grad():
            mask = batch["ori_mask"] if "ori_mask" in batch else None
            all_images = self._run_model_sample(batch["cond"], mask)

        all_images = torch.clamp(_to_pred01(all_images), 0.0, 1.0)
        nrow = 2 ** math.floor(math.log2(max(1.0, math.sqrt(batch["cond"].shape[0]))))
        tv.utils.save_image(all_images, str(self.results_folder / f"sample-{milestone}.png"), nrow=nrow)
        grid = tv.utils.make_grid(all_images, nrow=nrow)
        wandb.log({"samples": [wandb.Image(grid, caption=f"Sample Step {step}")]}, step=step)
        self.model.train()

    @staticmethod
    def _select_eval_metric(flat_metrics: Dict[str, float], metric_key: str, log_prefix: str) -> Optional[float]:
        key = metric_key.strip()
        if key.startswith(f"{log_prefix}/"):
            key = key[len(log_prefix) + 1 :]
        if "/" in key:
            key = key.split("/", 1)[1]
        if key in flat_metrics:
            return float(flat_metrics[key])
        return None

    def _run_evaluation(self, step: int, reason: str, target: Dict[str, Any]) -> Dict[str, float]:
        if not self.eval_enabled or target is None:
            return {}

        target_name = str(target.get("name", "val"))
        log_prefix = str(target.get("log_prefix", target_name))
        eval_loader = target.get("loader")
        eval_image_dir = Path(target.get("image_dir"))
        eval_gt_dir = Path(target.get("gt_dir"))
        eval_num_batches = int(target.get("num_batches", -1))
        eval_mode = str(target.get("mode", "edge"))
        eval_thresholds = target.get("thresholds", 99)
        eval_max_dist = float(target.get("max_dist", 0.0075))
        eval_apply_nms = bool(target.get("apply_nms", False))
        eval_apply_thinning = bool(target.get("apply_thinning", False))
        eval_nproc = int(target.get("nproc", 4))
        eval_preview_limit = int(target.get("preview_limit", 12))
        eval_select_metric = str(target.get("select_best_metric", "generic/AP"))

        if eval_loader is None:
            raise RuntimeError(f"Eval target '{target_name}' has no loader")
        if not eval_image_dir.is_dir():
            raise FileNotFoundError(f"Eval target '{target_name}' image_dir not found: {eval_image_dir}")
        if not eval_gt_dir.is_dir():
            raise FileNotFoundError(f"Eval target '{target_name}' gt_dir not found: {eval_gt_dir}")

        self.model.eval()
        device = self.accelerator.device

        step_dir = self.eval_output_root / f"{target_name}_step_{step:07d}"
        preds_dir = step_dir / "preds"
        gt_dir = step_dir / "gt"
        previews_dir = step_dir / "previews"
        preds_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)

        wandb_samples: List[Dict[str, Any]] = []
        exported = 0

        with torch.no_grad():
            iterator = tqdm(eval_loader, desc=f"{target_name}@{step} ({reason})", leave=False)
            for batch_idx, batch in enumerate(iterator):
                if eval_num_batches > 0 and batch_idx >= eval_num_batches:
                    break

                cond = batch["cond"].to(device, non_blocking=True)
                mask = batch["ori_mask"].to(device, non_blocking=True) if "ori_mask" in batch else None

                # Keep sampling path numerically aligned with training-time model assumptions:
                # run diffusion at configured model resolution, then map predictions back to raw size.
                model_h, model_w = int(self.image_size[0]), int(self.image_size[1])
                if cond.shape[-2:] != (model_h, model_w):
                    cond_for_model = F.interpolate(cond, size=(model_h, model_w), mode="bilinear", align_corners=True)
                else:
                    cond_for_model = cond
                if mask is not None and mask.shape[-2:] != (model_h, model_w):
                    mask_for_model = F.interpolate(mask, size=(model_h, model_w), mode="nearest")
                else:
                    mask_for_model = mask

                pred_batch = self._run_model_sample(cond_for_model, mask_for_model)

                names = batch["img_name"]
                if isinstance(names, str):
                    names = [names]
                elif isinstance(names, tuple):
                    names = list(names)

                raw_sizes = self._extract_raw_sizes(batch.get("raw_size"), len(names))

                if pred_batch.shape[0] != len(names):
                    raise RuntimeError(
                        f"Prediction batch size mismatch: preds={pred_batch.shape[0]}, names={len(names)}"
                    )

                for item_idx, item_name in enumerate(names):
                    stem = Path(item_name).stem
                    pred = _to_pred01(pred_batch[item_idx])

                    raw_w, raw_h = raw_sizes[item_idx]
                    if raw_w > 0 and raw_h > 0 and pred.shape[-2:] != (raw_h, raw_w):
                        pred = F.interpolate(
                            pred.unsqueeze(0),
                            size=(raw_h, raw_w),
                            mode="bilinear",
                            align_corners=True,
                        ).squeeze(0)
                        pred = torch.clamp(pred, 0.0, 1.0)

                    gt_path = self._find_gt_path_in_dir(eval_gt_dir, stem)
                    if gt_path is None:
                        gt_hint = "trainer.eval.gt_dir" if target_name == "val" else f"trainer.eval.{target_name}.gt_dir"
                        raise FileNotFoundError(
                            f"Missing GT for '{stem}' in {eval_gt_dir}. "
                            f"Set {gt_hint} or fix dataset alignment."
                        )

                    gt = _load_gt01(gt_path)
                    if gt.shape[-2:] != pred.shape[-2:]:
                        gt = F.interpolate(gt.unsqueeze(0), size=pred.shape[-2:], mode="nearest").squeeze(0)

                    pred_path = preds_dir / f"{stem}.png"
                    gt_out_path = gt_dir / f"{stem}.png"
                    _save_gray01_png(pred_path, pred)
                    _save_gray01_png(gt_out_path, gt)
                    exported += 1

                    if len(wandb_samples) < eval_preview_limit:
                        wandb_samples.append(
                            {
                                "id": stem,
                                "input": cond[item_idx].detach().cpu(),
                                "gt": gt.detach().cpu(),
                                "pred": pred.detach().cpu(),
                            }
                        )

        if exported == 0:
            raise RuntimeError("Evaluation exported zero samples. Check eval dataset and configuration.")

        eval_result = evaluate_dataset(
            gt_dir=str(gt_dir),
            pred_dir=str(preds_dir),
            mode=eval_mode,
            thresholds=eval_thresholds,
            max_dist=eval_max_dist,
            apply_nms=eval_apply_nms,
            apply_thinning=eval_apply_thinning,
            nproc=eval_nproc,
            save_dir=str(step_dir / "edgeeval_json") if eval_mode == "edge" else None,
        )
        flat_metrics = flatten_metrics(eval_result)

        preview_paths = make_previews(
            img_dir=str(eval_image_dir),
            pred_dir=str(preds_dir),
            gt_dir=str(gt_dir),
            out_dir=str(previews_dir),
            limit=eval_preview_limit,
        )

        selected_score = self._select_eval_metric(flat_metrics, eval_select_metric, log_prefix)
        if selected_score is not None and selected_score > self.best_eval_scores.get(target_name, float("-inf")):
            self.best_eval_scores[target_name] = selected_score
            self.best_eval_steps[target_name] = step

        with open(step_dir / "eval_results.json", "w", encoding="utf-8") as handle:
            json.dump(_jsonify(eval_result), handle, indent=2)
        with open(step_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(_jsonify(flat_metrics), handle, indent=2)

        if wandb_samples and target_name == "val":
            pred_grid = torch.stack([sample["pred"] for sample in wandb_samples], dim=0)
            milestone = step // self.save_and_sample_every if self.save_and_sample_every > 0 else step
            nrow = 2 ** math.floor(math.log2(max(1.0, math.sqrt(pred_grid.shape[0]))))
            tv.utils.save_image(pred_grid, str(self.results_folder / f"sample-{milestone}.png"), nrow=nrow)

        log_dict: Dict[str, Any] = {f"{log_prefix}/{k}": v for k, v in flat_metrics.items()}
        log_dict[f"{log_prefix}/exported_samples"] = exported
        log_dict[f"{log_prefix}/preview_count"] = len(preview_paths)
        if selected_score is not None:
            log_dict[f"{log_prefix}/selected_metric"] = selected_score
            log_dict[f"{log_prefix}/best_selected_metric"] = self.best_eval_scores.get(target_name, float("-inf"))
            log_dict[f"{log_prefix}/best_selected_metric_step"] = self.best_eval_steps.get(target_name, -1)

        if wandb_samples:
            triplets: List[Any] = []
            for sample in wandb_samples:
                input_img = _tensor_to_wandb_image(sample["input"], denorm=True)
                gt_img = _tensor_to_wandb_image(sample["gt"])
                pred_img = _tensor_to_wandb_image(sample["pred"])
                triplet = np.concatenate([input_img, gt_img, pred_img], axis=1)
                triplets.append(wandb.Image(triplet, caption=sample["id"]))
            log_dict[f"{log_prefix}/triplets"] = triplets

        if preview_paths:
            log_dict[f"{log_prefix}/previews"] = [wandb.Image(path) for path in preview_paths]

        wandb.log(log_dict, step=step)
        self.logger.info(
            f"[Eval:{target_name}] step={step} reason={reason} "
            + " ".join(f"{k}={v:.4f}" for k, v in flat_metrics.items())
        )

        self.model.train()
        return flat_metrics

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.eval_enabled and self.eval_at_step0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                reason = "step0" if self.step == 0 else "resume_preflight"
                for target in self.eval_targets:
                    self._run_evaluation(step=self.step, reason=reason, target=target)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                total_loss_dict = {"loss_simple": 0.0, "loss_vlb": 0.0, "total_loss": 0.0, "lr": 5e-5}

                for ga_ind in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(device, non_blocking=True)

                    if self.step == 0 and ga_ind == 0:
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            self.model.module.on_train_batch_start(batch)
                        else:
                            self.model.on_train_batch_start(batch)

                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict = self.model.module.training_step(batch)
                        else:
                            loss, log_dict = self.model.training_step(batch)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        loss_simple = log_dict["train/loss_simple"].item() / self.gradient_accumulate_every
                        loss_vlb = log_dict["train/loss_vlb"].item() / self.gradient_accumulate_every
                        total_loss_dict["loss_simple"] += loss_simple
                        total_loss_dict["loss_vlb"] += loss_vlb
                        total_loss_dict["total_loss"] += total_loss

                    self.accelerator.backward(loss)

                total_loss_dict["lr"] = self.opt.param_groups[0]["lr"]
                description = dict2str(total_loss_dict)
                description = f"[Train Step] {self.step}/{self.train_num_steps}: " + description
                if accelerator.is_main_process:
                    pbar.desc = description

                if self.step % self.log_freq == 0 and accelerator.is_main_process:
                    self.logger.info(description)

                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.lr_scheduler.step()

                if accelerator.is_main_process:
                    self.writer.add_scalar("Learning_Rate", self.opt.param_groups[0]["lr"], self.step)
                    self.writer.add_scalar("total_loss", total_loss, self.step)
                    self.writer.add_scalar("loss_simple", loss_simple, self.step)
                    self.writer.add_scalar("loss_vlb", loss_vlb, self.step)

                    wandb.log(
                        {
                            "train/learning_rate": self.opt.param_groups[0]["lr"],
                            "train/total_loss": total_loss,
                            "train/loss_simple": loss_simple,
                            "train/loss_vlb": loss_vlb,
                        },
                        step=self.step,
                    )

                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        if total_loss < self.best_loss:
                            previous_best = self.best_loss
                            self.best_loss = total_loss
                            self.logger.info(
                                f"Improvement found (Loss: {previous_best:.5f} -> {self.best_loss:.5f}). "
                                "Saving model-best.pt"
                            )
                            self.save("best")
                            wandb.log({"train/best_loss": self.best_loss}, step=self.step)
                        else:
                            self.logger.info(
                                f"No improvement (Loss: {total_loss:.5f} >= Best: {self.best_loss:.5f}). "
                                "Skipping save."
                            )

                        if not self.eval_enabled:
                            self._save_batch_sample(batch=batch, milestone=milestone, step=self.step)

                if self.eval_enabled and self.step % self.eval_every_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        for target in self.eval_targets:
                            self._run_evaluation(step=self.step, reason="periodic", target=target)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()

                pbar.update(1)

        if accelerator.is_main_process:
            wandb.finish()

        accelerator.print("training complete")


if __name__ == "__main__":
    args = parse_args()
    main(args)
