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

## API Arguments
| Name | Default | Description |
|---|---|---|
| `gt_dir` | required | Directory of GT maps. |
| `pred_dir` | required | Directory of prediction maps. |
| `mode` | `binary` | `binary` for generic metrics only, `edge` for pyEdgeEval + generic. |
| `thresholds` | `99` | Threshold spec used by pyEdgeEval in `edge` mode. |
| `max_dist` | `0.0075` | Edge matching tolerance in `edge` mode. |
| `apply_thinning` | `False` | Enable thinning in `edge` mode. |
| `apply_nms` | `False` | Enable NMS in `edge` mode. |
| `nproc` | `4` | Worker count for pyEdgeEval. |
| `save_dir` | `None` | Optional pyEdgeEval result output directory. |

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

