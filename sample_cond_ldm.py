import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from fvcore.common.config import CfgNode
from scipy import integrate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.data import EdgeDatasetTest
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.utils import unnormalize_to_zero_to_one
from eval.eval import evaluate_dataset, flatten_metrics
from eval.previews import make_previews
from eval.run_params import save_run_params


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling / evaluation config")
    parser.add_argument("--cfg", help="experiment config file", type=str, required=True)

    # Optional eval overrides
    parser.add_argument("--gt_dir", type=str, default=None, help="GT directory for evaluation")
    parser.add_argument("--img_dir", type=str, default=None, help="Input image directory for previews")
    parser.add_argument("--eval_mode", type=str, choices=["binary", "edge"], default=None)
    parser.add_argument("--eval_thresholds", type=str, default=None)
    parser.add_argument("--eval_max_dist", type=float, default=None)
    parser.add_argument("--eval_apply_nms", action="store_true")
    parser.add_argument("--eval_apply_thinning", action="store_true")
    parser.add_argument("--eval_nproc", type=int, default=None)
    parser.add_argument("--eval_preview_limit", type=int, default=None)
    parser.add_argument("--eval_out_dir", type=str, default=None)

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


def _cfg_to_dict(value: Any) -> Any:
    if isinstance(value, CfgNode):
        return {k: _cfg_to_dict(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _cfg_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cfg_to_dict(v) for v in value]
    return value


def _resolve_eval_settings(args, cfg: CfgNode, pred_dir: Path) -> Optional[Dict[str, Any]]:
    sampler_cfg = cfg.sampler
    data_cfg = cfg.data

    eval_enabled_cfg = bool(sampler_cfg.get("eval_enabled", False))
    eval_requested_cli = any(
        value is not None
        for value in [
            args.gt_dir,
            args.eval_mode,
            args.eval_thresholds,
            args.eval_max_dist,
            args.eval_nproc,
            args.eval_preview_limit,
            args.eval_out_dir,
            args.img_dir,
        ]
    ) or args.eval_apply_nms or args.eval_apply_thinning

    if not eval_enabled_cfg and not eval_requested_cli:
        return None

    gt_dir = args.gt_dir or sampler_cfg.get("eval_gt_dir")
    if not gt_dir:
        raise ValueError("Evaluation requested but GT directory is not provided. Use --gt_dir or sampler.eval_gt_dir.")

    img_dir = args.img_dir or sampler_cfg.get("eval_img_dir") or data_cfg.img_folder
    out_dir = Path(args.eval_out_dir or sampler_cfg.get("eval_out_dir") or (pred_dir / "eval"))

    eval_mode = args.eval_mode or sampler_cfg.get("eval_mode", "edge")
    eval_thresholds = args.eval_thresholds if args.eval_thresholds is not None else sampler_cfg.get("eval_thresholds", 99)
    eval_max_dist = float(args.eval_max_dist if args.eval_max_dist is not None else sampler_cfg.get("eval_max_dist", 0.0075))
    eval_apply_nms = bool(args.eval_apply_nms or sampler_cfg.get("eval_apply_nms", False))
    eval_apply_thinning = bool(args.eval_apply_thinning or sampler_cfg.get("eval_apply_thinning", False))
    eval_nproc = int(args.eval_nproc if args.eval_nproc is not None else sampler_cfg.get("eval_nproc", 4))
    eval_preview_limit = int(
        args.eval_preview_limit if args.eval_preview_limit is not None else sampler_cfg.get("eval_preview_limit", 24)
    )

    gt_dir_path = Path(gt_dir)
    img_dir_path = Path(img_dir)
    if not gt_dir_path.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir_path}")
    if not img_dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir_path}")

    return {
        "gt_dir": gt_dir_path,
        "img_dir": img_dir_path,
        "out_dir": out_dir,
        "mode": eval_mode,
        "thresholds": eval_thresholds,
        "max_dist": eval_max_dist,
        "apply_nms": eval_apply_nms,
        "apply_thinning": eval_apply_thinning,
        "nproc": eval_nproc,
        "preview_limit": eval_preview_limit,
    }


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)

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
        raise NotImplementedError

    dataset = EdgeDatasetTest(
        data_root=data_cfg.img_folder,
        image_size=model_cfg.image_size,
    )
    dl = DataLoader(
        dataset,
        batch_size=cfg.sampler.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=data_cfg.get("num_workers", 2),
    )

    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm,
        dl,
        batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder,
        cfg=cfg,
    )

    save_run_params(_cfg_to_dict(cfg), sampler.results_folder, prefix="sample")
    outputs = sampler.sample()
    if not outputs:
        raise RuntimeError("No predictions were produced during sampling.")

    eval_settings = _resolve_eval_settings(args=args, cfg=cfg, pred_dir=sampler.results_folder)
    if eval_settings is not None:
        eval_out_dir = eval_settings["out_dir"]
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        eval_result = evaluate_dataset(
            gt_dir=str(eval_settings["gt_dir"]),
            pred_dir=str(sampler.results_folder),
            mode=eval_settings["mode"],
            thresholds=eval_settings["thresholds"],
            max_dist=eval_settings["max_dist"],
            apply_nms=eval_settings["apply_nms"],
            apply_thinning=eval_settings["apply_thinning"],
            nproc=eval_settings["nproc"],
            save_dir=str(eval_out_dir / "edgeeval_json") if eval_settings["mode"] == "edge" else None,
        )
        flat = flatten_metrics(eval_result)

        preview_paths = make_previews(
            img_dir=str(eval_settings["img_dir"]),
            pred_dir=str(sampler.results_folder),
            gt_dir=str(eval_settings["gt_dir"]),
            out_dir=str(eval_out_dir / "previews"),
            limit=eval_settings["preview_limit"],
        )
        if len(preview_paths) == 0:
            print(
                "[WARN] Preview generation produced 0 files. "
                "Check stem overlap and that GT/pred/input directories contain supported map/image formats."
            )

        with open(eval_out_dir / "eval_results.json", "w", encoding="utf-8") as handle:
            json.dump(_jsonify(eval_result), handle, indent=2)
        with open(eval_out_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(_jsonify(flat), handle, indent=2)

        print("=" * 60)
        print("EVAL METRICS")
        for key, value in sorted(flat.items()):
            print(f"  {key}: {value:.6f}")
        print(f"previews: {len(preview_paths)}")
        print(f"results: {eval_out_dir}")
        print("=" * 60)


class Sampler(object):
    def __init__(
        self,
        model,
        data_loader,
        sample_num=1000,
        batch_size=16,
        results_folder="./results",
        rk45=False,
        cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision="no",
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45
        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size) if sample_num > 0 else -1
        self.image_size = model.image_size
        self.cfg = cfg

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        ckpt_path = Path(cfg.sampler.ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        data = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
        model = self.accelerator.unwrap_model(self.model)

        if cfg.sampler.use_ema:
            if "ema" not in data:
                raise KeyError("sampler.use_ema=True but checkpoint has no 'ema' key")
            sd = data["ema"]
            new_sd = {}
            for key in sd.keys():
                if key.startswith("ema_model."):
                    new_sd[key[10:]] = sd[key]
            if not new_sd:
                raise KeyError("EMA checkpoint found but no 'ema_model.*' weights were extracted")
            model.load_state_dict(new_sd)
        else:
            if "model" not in data:
                raise KeyError("Checkpoint is missing required 'model' key")
            model.load_state_dict(data["model"])

        if "model" in data and "scale_factor" in data["model"]:
            model.scale_factor = data["model"]["scale_factor"]

    def sample(self) -> List[str]:
        accelerator = self.accelerator
        device = accelerator.device
        outputs: List[str] = []

        with torch.no_grad():
            self.model.eval()
            for _, batch in tqdm(enumerate(self.dl), desc="sampling"):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device, non_blocking=True)

                cond = batch["cond"]
                raw_w = batch["raw_size"][0].item()
                raw_h = batch["raw_size"][1].item()
                img_names = batch["img_name"]
                if isinstance(img_names, str):
                    img_names = [img_names]
                elif isinstance(img_names, tuple):
                    img_names = list(img_names)
                mask = batch["ori_mask"] if "ori_mask" in batch else None

                if self.cfg.sampler.sample_type == "whole":
                    batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
                elif self.cfg.sampler.sample_type == "slide":
                    batch_pred = self.slide_sample(
                        cond,
                        crop_size=self.cfg.sampler.get("crop_size", [320, 320]),
                        stride=self.cfg.sampler.stride,
                        mask=mask,
                    )
                else:
                    raise NotImplementedError(f"Unsupported sample_type: {self.cfg.sampler.sample_type}")

                for index, pred in enumerate(batch_pred):
                    img_name = img_names[index] if index < len(img_names) else img_names[0]
                    file_name = self.results_folder / img_name
                    out_path = str(file_name)[:-4] + ".png"
                    tv.utils.save_image(pred, out_path)
                    outputs.append(out_path)

        accelerator.print("sampling complete")
        return outputs

    def slide_sample(self, inputs, crop_size, stride, mask=None):
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        model_h, model_w = self.cfg.model.image_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                crop_h = int(y2 - y1)
                crop_w = int(x2 - x1)

                if crop_h != model_h or crop_w != model_w:
                    model_input = F.interpolate(crop_img, size=(model_h, model_w), mode="bilinear", align_corners=True)
                else:
                    model_input = crop_img

                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(
                        batch_size=model_input.shape[0], cond=model_input, mask=mask
                    )
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample(
                        batch_size=model_input.shape[0], cond=model_input, mask=mask
                    )
                else:
                    raise NotImplementedError

                if crop_h != model_h or crop_w != model_w:
                    crop_seg_logit = F.interpolate(
                        crop_seg_logit, size=(crop_h, crop_w), mode="bilinear", align_corners=True
                    )

                preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        if int((count_mat == 0).sum()) > 0:
            raise RuntimeError("Invalid sliding window accumulation: found zero-count pixels")

        return preds / count_mat

    def whole_sample(self, inputs, raw_size, mask=None):
        inputs = F.interpolate(inputs, size=(416, 416), mode="bilinear", align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        else:
            raise NotImplementedError

        return F.interpolate(seg_logits, size=raw_size, mode="bilinear", align_corners=True)

    def cal_fid(self, target_path):
        command = "fidelity -g 0 -f -i -b {} --input1 {} --input2 {}".format(
            self.batch_size, str(self.results_folder), target_path
        )
        os.system(command)

    def rk45_sample(self, batch_size):
        with torch.no_grad():
            shape = (batch_size, 3, *(self.image_size))
            ode_sampler = get_ode_sampler(method="RK45")
            x, nfe = ode_sampler(model=self.model, shape=shape)
            x = unnormalize_to_zero_to_one(x)
            x.clamp_(0.0, 1.0)
            return x, nfe


def get_ode_sampler(rtol=1e-5, atol=1e-5, method="RK45", eps=1e-3, device="cuda"):
    def drift_fn(model, x, t, model_type="const"):
        pred = model(x, t)
        if model_type == "const":
            drift = pred
        elif model_type == "linear":
            k, c = pred.chunk(2, dim=1)
            drift = k * t + c
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return drift

    def ode_sampler(model, shape):
        with torch.no_grad():
            x = torch.randn(*shape)

            def ode_func(t, x_flat):
                x_tensor = from_flattened_numpy(x_flat, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x_tensor.device) * t * 1000
                drift = drift_fn(model, x_tensor, vec_t)
                return to_flattened_numpy(drift)

            solution = integrate.solve_ivp(
                ode_func,
                (1, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            nfe = solution.nfev
            sample = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            return sample, nfe

    return ode_sampler


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


if __name__ == "__main__":
    args = parse_args()
    main(args)
