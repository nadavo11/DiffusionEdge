## DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection ([arXiv](https://arxiv.org/abs/2401.02032))
[Yunfan Ye](https://yunfan1202.github.io), [Yuhang Huang](https://github.com/GuHuangAI), [Renjiao Yi](https://renjiaoyi.github.io/), [Zhiping Cai](), [Kai Xu](http://kevinkaixu.net/index.html)

![Teaser](assets/teaser.png)
![](assets/denoising_process/3063/test.gif)
![](assets/denoising_process/5096/test.gif)

This README is rewritten for the current code in this repository, with special focus on training and evaluation workflows for **RWTD** and custom datasets such as **seams**.

## 1. Repository Entry Points

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `train_vae.py` | Train first-stage autoencoder (`AutoencoderKL`) | `--cfg` (YAML) | Checkpoints + preview images in `trainer.results_folder` |
| `train_cond_ldm.py` | Train latent diffusion edge model (`LatentDiffusion`) | `--cfg` (YAML) | `model-best.pt`, samples, TensorBoard, logs, WandB |
| `sample_cond_ldm.py` | Config-driven inference over an image directory | `--cfg` (YAML) | Predicted edge maps in `sampler.save_folder` |
| `demo.py` | CLI inference without editing checkpoint path inside YAML | `--input_dir`, `--pre_weight`, `--out_dir` (+ optional args) | Predicted edge maps in `--out_dir` |
| `demo_trt.py` | TensorRT runtime inference | TRT engine + input images | Predicted edge maps in `--out_dir` |

## 2. Environment Setup

### 2.1 Python and PyTorch
```bash
conda create -n diffedge python=3.9
conda activate diffedge
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

### 2.2 Project dependencies
```bash
pip install -r requirement.txt
pip install wandb
```

`train_cond_ldm.py` imports and uses `wandb` directly, so install it explicitly.

### 2.3 Accelerate setup (required for `accelerate launch ...`)
```bash
accelerate config
```

### 2.4 Optional environment variables

| Variable | Required | Why |
|---|---|---|
| `WANDB_API_KEY` | No | Needed only if you want online WandB logging |
| `WANDB_MODE=offline` | No | Use this if you want local-only WandB runs |
| `CUDA_VISIBLE_DEVICES` | No | Select visible GPUs |

## 3. Data Layout Requirements (Critical)

The code has **two different dataset contracts**.

### 3.1 Training contract (`EdgeDataset` in `denoising_diffusion_pytorch/data.py`)

`train_cond_ldm.py` supports both of these dataset roots:

Split layout (existing datasets like BSDS):

```text
<DATA_ROOT>/
  image/
    train/
      0001.jpg
      0002.jpg
    val/
      ...
    test/
      ...
  edge/
    train/
      0001.jpg or 0001.png
      0002.jpg or 0002.png
    val/
      ...
    test/
      ...
```

Flat layout (synthetic texture style):

```text
<DATA_ROOT>/
  images/
    000001.png
    000002.png
  edges/
    000001.png
    000002.png
```

Important loader invariants from code:
- It supports both `image/edge` and `images/edges` roots.
- In split mode, it enumerates `image/*` subdirectories (`train`, `val`, `test`, etc.) and pairs files by basename.
- In flat mode, it pairs files directly under `images/` and `edges/` by basename.
- Pairing logic only auto-falls back between `.jpg` and `.png` (`fit_img_postfix`).
- Edge `.mat` files are **not** supported by `EdgeDataset.read_lb` (PIL image read is used).
- Returned tensors:
  - `cond`: shape `(B, 3, H, W)`, `float32`, normalized to `[-1, 1]`
  - `image`: shape `(B, 1, H, W)`, `float32`, normalized to `[-1, 1]`
- `H, W` come from `model.image_size` after crop/resize transform.

### 3.2 Inference/evaluation contract (`EdgeDatasetTest`)

`sample_cond_ldm.py` and `demo.py` expect a **flat image directory** (not nested):

```text
<EVAL_IMAGE_DIR>/
  a.jpg
  b.jpg
  c.png
```

Supported extensions are: `.jpg`, `.JPG`, `.png`, `.pgm`, `.ppm`.

## 4. RWTD Workflow

### 4.1 Current state of this local repo snapshot

Observed under `RWTD/`:
- `RWTD/image/train` and `RWTD/image/val` contain `.jpg` files.
- `RWTD/edge/train` and `RWTD/edge/val` contain `.mat` files.
- `RWTD/image/test` and `RWTD/edge/test` are currently empty.

This means RWTD labels need conversion to image files before training with `EdgeDataset`.

### 4.2 Convert RWTD edge labels from `.mat` to `.jpg`

This repo already contains a conversion utility at `BSDS/edge/train/mat2jpg.py`.
One practical way:

```bash
for split in train val test; do
  [ -d "RWTD/edge/$split" ] || continue
  cp BSDS/edge/train/mat2jpg.py "RWTD/edge/$split/mat2jpg.py"
  (cd "RWTD/edge/$split" && python mat2jpg.py)
done
```

After conversion, check that image and edge basenames align:

```bash
comm -3 \
  <(ls RWTD/image/train | sed 's/\.[^.]*$//' | sort) \
  <(ls RWTD/edge/train  | sed 's/\.[^.]*$//' | sort) | head
```

This command should print nothing when pairs are aligned.

### 4.3 Train RWTD model

Create a RWTD train config from `configs/BSDS_train.yaml`:

```bash
cp configs/BSDS_train.yaml configs/RWTD_train.yaml
```

Edit at least:
- `data.img_folder: 'RWTD'`
- `trainer.results_folder: './training/RWTD_swin_unet12_disloss_bs2x8'`
- `model.first_stage.ckpt_path`: set to your first-stage checkpoint if you have one

Then launch:

```bash
accelerate launch train_cond_ldm.py --cfg configs/RWTD_train.yaml
```

Training writes `model-best.pt` to `trainer.results_folder` when loss improves.

### 4.4 Evaluate RWTD (generate predictions)

Create a RWTD sampling config:

```bash
cp configs/RWTD_sample.yaml configs/RWTD_val_sample.yaml
```

Edit at least:
- `data.img_folder: 'RWTD/image/val'` (or your eval split path)
- `sampler.ckpt_path: './training/RWTD_swin_unet12_disloss_bs2x8/model-best.pt'`
- `sampler.save_folder: './out_rwtd'`
- Keep `sampler.sample_type: slide` and `sampler.batch_size: 1`

Run:

```bash
python sample_cond_ldm.py --cfg configs/RWTD_val_sample.yaml
```

Output naming pattern:
- Input `123.jpg` -> output `123.png` in `sampler.save_folder`.

### 4.5 Cross-domain: Train on synthetic texture, evaluate on RWTD

This repository now includes ready-to-use configs for that exact flow:
- `configs/synthetic_texture_train.yaml`
- `configs/synthetic_texture_to_RWTD_sample.yaml`

Train on synthetic texture:

```bash
accelerate launch train_cond_ldm.py --cfg configs/synthetic_texture_train.yaml
```

Evaluate the trained checkpoint on RWTD validation images:

```bash
python sample_cond_ldm.py --cfg configs/synthetic_texture_to_RWTD_sample.yaml
```

If needed, edit:
- `model.first_stage.ckpt_path` in both files.
- `sampler.ckpt_path` in `configs/synthetic_texture_to_RWTD_sample.yaml`.
- `data.img_folder` in `configs/synthetic_texture_to_RWTD_sample.yaml` for a different RWTD split.

## 5. seams Workflow (Custom Dataset)

No seams-specific config is shipped in this repo, but seams works with the same contract.

### 5.1 Required seams layout

```text
<SEAMS_ROOT>/
  image/
    train/
    val/
    test/      # optional
  edge/
    train/
    val/
    test/      # optional
```

File pairing is basename-based (same rules as RWTD section).

### 5.2 Train on seams

```bash
cp configs/BSDS_train.yaml configs/seams_train.yaml
```

Edit at least:
- `data.img_folder: '<SEAMS_ROOT>'`
- `trainer.results_folder: './training/seams_swin_unet12_disloss_bs2x8'`
- `model.first_stage.ckpt_path`: set correctly for your run

Then:

```bash
accelerate launch train_cond_ldm.py --cfg configs/seams_train.yaml
```

### 5.3 Evaluate seams

```bash
cp configs/RWTD_sample.yaml configs/seams_sample.yaml
```

Edit at least:
- `data.img_folder: '<SEAMS_ROOT>/image/val'` (flat eval directory for `EdgeDatasetTest`)
- `sampler.ckpt_path: './training/seams_swin_unet12_disloss_bs2x8/model-best.pt'`
- `sampler.save_folder: './out_seams'`

Then:

```bash
python sample_cond_ldm.py --cfg configs/seams_sample.yaml
```

## 6. CLI Reference

### 6.1 `train_vae.py`

| Name | Default | Type | Description |
|---|---|---|---|
| `--cfg` | None (required) | `str` | YAML config path for first-stage VAE training |

### 6.2 `train_cond_ldm.py`

| Name | Default | Type | Description |
|---|---|---|---|
| `--cfg` | None (required) | `str` | YAML config path for latent diffusion training |

### 6.3 `sample_cond_ldm.py`

| Name | Default | Type | Description |
|---|---|---|---|
| `--cfg` | None (required) | `str` | YAML config path for sampling/inference |

### 6.4 `demo.py`

| Name | Default | Type | Description |
|---|---|---|---|
| `--cfg` | `./configs/default.yaml` | `str` | Base model/sampler config |
| `--input_dir` | None (required) | `str` | Flat directory containing input images |
| `--pre_weight` | None (required) | `str` | Checkpoint path |
| `--sampling_timesteps` | `1` | `int` | Number of diffusion sampling steps |
| `--out_dir` | None (required) | `str` | Output directory |
| `--bs` | `8` | `int` | Window inference micro-batch size inside `slide_sample` |

### 6.5 `demo_trt.py`

| Name | Default | Type | Description |
|---|---|---|---|
| `--input_dir` | None (required) | `str` | Flat directory containing input images |
| `--pre_weight` | None (required) | `str` | TensorRT engine path (`.trt`) |
| `--out_dir` | None (required) | `str` | Output directory |
| `--bs` | `16` | `int` | TRT inference batch size |
| `--crop_size` | `256` | `int` | Sliding crop size |

## 7. Config Surfaces You Actually Need

### 7.1 Training config (`train_cond_ldm.py`)

| Name | Default (from `configs/BSDS_train.yaml`) | Type | Description |
|---|---|---|---|
| `data.img_folder` | `'BSDS'` | `str` | Dataset root, must contain `image/*` and `edge/*` subdirs |
| `data.crop_type` | `rand_resize_crop` | `str` | `rand_crop` or `rand_resize_crop` |
| `data.batch_size` | `4` | `int` | Dataloader batch size |
| `data.num_workers` | `8` | `int` | Dataloader workers |
| `model.image_size` | `[320, 320]` | `list[int,int]` | Training crop/output size |
| `model.first_stage.ckpt_path` | `'./checkpoints/first_stage_total_320.pt'` | `str` | First-stage autoencoder weights |
| `trainer.gradient_accumulate_every` | `8` | `int` | Gradient accumulation steps |
| `trainer.lr` | `5e-5` | `float` | Optimizer LR |
| `trainer.min_lr` | `5e-6` | `float` | Scheduler floor |
| `trainer.train_num_steps` | `100000` | `int` | Total train steps |
| `trainer.save_and_sample_every` | `200` | `int` | Sampling/checkpoint interval |
| `trainer.results_folder` | `./training/BSDS_swin_unet12_disloss_bs2x8` | `str` | Output root |
| `trainer.enable_resume` | `True` | `bool` | Enables optimizer/scheduler/scaler checkpoint state |
| `trainer.resume_milestone` | `0` | `int` | Resume file suffix (`model-<milestone>.pt`) |
| `trainer.test_before` | `True` | `bool` | Save/log a pre-training sample at start |
| `trainer.ema_update_after_step` | `5000` | `int` | EMA warmup |
| `trainer.ema_update_every` | `10` | `int` | EMA update interval |

### 7.2 Sampling config (`sample_cond_ldm.py`)

| Name | Default (from `configs/RWTD_sample.yaml`) | Type | Description |
|---|---|---|---|
| `data.img_folder` | `'RWTD/image/train'` | `str` | Flat image directory to run inference on |
| `sampler.sample_type` | `slide` | `str` | `slide` or `whole` |
| `sampler.crop_size` | `[320, 320]` | `list[int,int]` | Tile size for sliding inference |
| `sampler.stride` | `[320, 320]` | `list[int,int]` | Tile stride |
| `sampler.batch_size` | `1` | `int` | Dataloader batch size in sampling script |
| `sampler.sample_num` | `-1` | `int` | Present in config, not used to limit loop for edge data path |
| `sampler.use_ema` | `False` | `bool` | Load `ema` weights from checkpoint if available |
| `sampler.save_folder` | `./results_rwtd` | `str` | Output folder |
| `sampler.ckpt_path` | `"checkpoints/my-best.pt"` | `str` | Model checkpoint path |

## 8. Outputs, Logging, and Metrics

### 8.1 Training outputs (`train_cond_ldm.py`)
- `trainer.results_folder/model-best.pt`:
  - Overwritten when a new best loss is found.
  - Contains model weights (`model`) and, when resume is enabled, optimizer/scheduler/EMA/scaler state.
- `trainer.results_folder/sample-<milestone>.png`: periodic visual samples.
- `trainer.results_folder/<timestamp>_.log`: logger output.
- `trainer.results_folder/events.out.tfevents...`: TensorBoard.
- WandB keys:
  - `train/learning_rate` (higher/lower: context dependent)
  - `train/loss_simple` (lower is better)
  - `train/loss_vlb` (lower is better)
  - `train/total_loss` (lower is better)
  - `train/best_loss` (lower is better)
  - image panels under `samples` and `test_before_samples`.

### 8.2 Inference outputs
- `sample_cond_ldm.py` and `demo.py` write one `.png` per input image basename.
- Re-running on the same output folder overwrites files with same names.
- Writes use direct `torch.save` / `tv.utils.save_image` calls (not atomic writes).

### 8.3 Quantitative evaluation
- For `data.name == edge`, `sample_cond_ldm.py` exits right after writing predictions.
- Built-in FID call in `sample_cond_ldm.py` is not used for edge mode.
- Typical edge metrics (ODS/OIS/SEval/CEval) should be run with your external evaluator after prediction export.

## 9. Troubleshooting

1. `FileNotFoundError` or PIL decode errors on labels during training  
Cause: edge labels are `.mat` or filename pairs do not match image basenames.  
Fix: convert labels to `.jpg/.png` and validate basename alignment.

2. `KeyError: 'ema'` when sampling  
Cause: `sampler.use_ema: True` but checkpoint has no `ema` state.  
Fix: set `sampler.use_ema: False` or load a checkpoint that contains EMA.

3. Bad or unstable sampling quality from a trained checkpoint  
Cause: sampling config model block differs from training config (especially `model.objective`).  
Fix: keep model architecture/objective/timestep settings consistent with the training run.

4. Resume not working as expected  
Cause: `resume_milestone` points to missing `model-<milestone>.pt`, or resume disabled.  
Fix: set `trainer.enable_resume: True` and use an existing milestone checkpoint file.

5. WandB login prompts/failures on remote machine  
Cause: WandB online mode without auth.  
Fix: export `WANDB_MODE=offline` or set `WANDB_API_KEY`.

6. CUDA OOM during sampling  
Cause: `--bs` (demo) or `sampler.batch_size`/`crop_size` too large.  
Fix: reduce these values.

7. No outputs generated in sampling  
Cause: wrong `data.img_folder` path or unsupported file extension.  
Fix: ensure the directory is flat and contains supported image extensions.

8. Training starts but quality is poor from the beginning  
Cause: missing/incorrect first-stage checkpoint (`model.first_stage.ckpt_path`).  
Fix: use a valid first-stage weight or explicitly train first stage first.

## 10. Real-time TensorRT Inference (Optional)

```bash
python demo_trt.py \
  --input_dir <image_dir> \
  --pre_weight <engine.trt> \
  --out_dir <output_dir> \
  --bs 16 \
  --crop_size 256
```

## Contact
If you have questions, contact: `huangai@nudt.edu.cn`.

## Thanks
Thanks to the base code [DDM-Public](https://github.com/GuHuangAI/DDM-Public).

## Citation
```bibtex
@inproceedings{ye2024diffusionedge,
  title={DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection},
  author={Yunfan Ye and Kai Xu and Yuhang Huang and Renjiao Yi and Zhiping Cai},
  year={2024},
  booktitle={AAAI}
}
```
