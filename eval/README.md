# DiffusionEdge Eval Module

## Overview
`DiffusionEdge/eval` contains the shared evaluation stack used by:
- `train_cond_ldm.py` for step-0 and periodic validation
- `sample_cond_ldm.py` for optional offline evaluation after prediction export

The pipeline is:
1. Export predictions to `preds/`
2. Export matched-resolution GT to `gt/`
3. Run `evaluate_dataset(...)`
4. Generate preview panels (`Input | GT | Pred | Overlay`)
5. Save metrics JSON and optional pyEdgeEval outputs

## Setup Prerequisites
- Python dependencies:
  - required: `numpy`, `opencv-python`, `scikit-learn`
  - optional for edge metrics: `pyEdgeEval` (required only when `mode=edge`)
- Directory invariants:
  - prediction and GT files are matched by filename stem
  - maps must be grayscale-like with values in `[0,1]` or `[0,255]`
  - in `mode=edge`, `.mat` GT uses per-annotator `Boundaries` maps (BSDS-style, no annotator averaging)

## API Arguments
| Name | Default | Description |
|---|---|---|
| `gt_dir` | required | Directory of GT maps. |
| `pred_dir` | required | Directory of prediction maps. |
| `mode` | `binary` | `binary` for generic metrics only, `edge` for pyEdgeEval + generic. |
| `thresholds` | `99` | Threshold spec used by pyEdgeEval in `edge` mode. |
| `max_dist` | `None` | Edge matching tolerance in `edge` mode; resolved by `protocol` when unset. |
| `apply_thinning` | `None` | Enable thinning in `edge` mode; resolved by `protocol` when unset. |
| `apply_nms` | `None` | Enable NMS in `edge` mode; resolved by `protocol` when unset. |
| `protocol` | `None` | Edge protocol: `legacy` (old defaults) or `bsds` (thinning on by default). |
| `ap_mode` | `bsds_interp` | AP integration mode: `bsds_interp`, `trapz`, or `voc_interp`. |
| `include_generic_metrics` | `True` | Compute/report generic pixel metrics. Set `False` in edge mode to save CPU. |
| `edge_keys_mode` | `full` | Edge metric payload shape: `full` (explicit + aliases) or `legacy_minimal` (legacy BSDS keys only). |
| `nproc` | `4` | Worker count for pyEdgeEval. |
| `save_dir` | `None` | Optional pyEdgeEval result output directory. |

## Output Metric Keys
- Generic (explicit): `generic/AP_pixel`, `generic/ROC_AUC_pixel`, `generic/N_pixels`
- Edge (explicit): `edge/AP_pr`, `edge/AUC_pr`, `edge/ODS_f1`, `edge/OIS_pooled_f1`, `edge/OIS_macro_meanF1`
- Legacy aliases are still emitted for compatibility (`generic/AP`, `edge/AP`, `edge/OIS_f1`, etc.).
- If `edge_keys_mode=legacy_minimal`, edge output is reduced to:
  `edge/AP`, `edge/ODS_threshold`, `edge/ODS_recall`, `edge/ODS_precision`, `edge/ODS_f1`,
  `edge/OIS_recall`, `edge/OIS_precision`, `edge/OIS_f1`.

## Quick Start
Train with built-in eval (step-0 + periodic):
```bash
accelerate launch train_cond_ldm.py --cfg configs/synthetic_texture_train.yaml
```

Run inference + eval:
```bash
python sample_cond_ldm.py --cfg configs/synthetic_texture_to_RWTD_sample.yaml
```

Run inference only:
```bash
python demo.py --input_dir <images> --pre_weight <ckpt> --out_dir <pred_out>
```
