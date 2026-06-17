# SwissImage Pedestrian-Crossing Classifier

PyTorch scaffold for the hackathon task: classify RGB SwissTopo/SwissImage 25 m x 25 m patches as containing a pedestrian crossing or not.

At 10 cm/pixel, each patch should be `250 x 250` pixels:

```text
25 m / 0.10 m = 250 pixels per side
250 * 250 = 62,500 RGB pixels
```

The default training resolution is therefore `250`, preserving the native 10 cm/pixel scale. If your exported patches are actually `2500 x 2500`, that corresponds to 1 cm/pixel and should be handled with tiling or multi-instance learning rather than this simple full-image classifier.

## What This Uses

- Primary backbone: TorchGeo `ResNet50_Weights.FMOW_RGB_GASSL`, loaded into a `timm` ResNet-50 with the classifier removed.
- Custom head: global average pooling plus an aggressive MLP classifier. For ResNet-50 this is `2048 -> 2048 -> 256 -> 32 -> 2`, producing raw logits for `no` and `yes`.
- Training stages: adaptive frozen-backbone MLP warmup, `layer4` fine-tuning, optional full fine-tuning.

Tool/source disclosure for the model choice: the assignment PDF was inspected locally, and the external references used were TorchGeo pretrained weights, TorchGeo weight-loading examples, `timm` feature extraction docs, and SatlasPretrain model docs.

## Install

```powershell
uv sync --group dev
```

The first pretrained run will download TorchGeo/Hugging Face weights. For an offline shape check, use the smoke test without pretrained weights.

## Dataset Layout

Use two image folders with exactly two class subfolders:

```text
data/
  train/
    crossing/
    no_crossing/
  val/
    crossing/
    no_crossing/
```

The label is remapped so `crossing` is always positive (`1`) and the other folder is negative (`0`), independent of alphabetical folder order.

Prefer geographic splits when creating `train` and `val`: nearby SwissImage tiles can look nearly identical, and random patch-level splitting can overestimate validation performance.

## Prepare Dataset From Classmate Metadata

With permission, we can reuse only the public dataset metadata and Swisstopo reconstruction path from Oliver Schütz / DotNaos' repository. This does not use his model code.

Small smoke export:

```powershell
uv run crossing-prepare-classmate-data `
  --positive-limit 20 `
  --negative-ratio 1.0 `
  --image-size 250 `
  --overwrite
```

Larger balanced export:

```powershell
uv run crossing-prepare-classmate-data `
  --positive-limit 2500 `
  --negative-ratio 1.0 `
  --image-size 250 `
  --overwrite
```

This downloads the released metadata, reconstructs/caches required Swisstopo scene mosaics, and writes our classifier layout directly under `data/`:

```text
data/
  train/crossing/
  train/no_crossing/
  val/crossing/
  val/no_crossing/
  test/crossing/
  test/no_crossing/
```

It also writes `data/classmate_dataset_manifest.csv` and `data/classmate_dataset_summary.json` for disclosure and reproducibility.

## Import Own Zip Dataset

Sven's own dataset zip is expected at `data/svenzebradata.zip` with this structure:

```text
zebra/
  y/  # crossing
  n/  # no_crossing
```

Import it into the same `ImageFolder` layout:

```powershell
uv run crossing-import-own-data
```

The importer maps `zebra/y` to `crossing`, maps `zebra/n` to `no_crossing`, uses a deterministic `80/10/10` split, prefixes imported filenames with `own_svenzebradata_`, and writes:

- `data/own_dataset_summary.json`
- `data/own_dataset_manifest.csv`

Current own-data import:

| Split | crossing | no_crossing |
| --- | ---: | ---: |
| Train | 14 | 40946 |
| Validation | 2 | 5118 |
| Test | 1 | 5119 |

After merging with the classmate dataset, the local dataset is heavily imbalanced toward `no_crossing`. Keep `--pos-weight auto` enabled, and consider weighted sampling or negative downsampling before the final reported run.

## Smoke Test

No weight download:

```powershell
uv run crossing-smoke --no-pretrained
```

With the real pretrained backbone:

```powershell
uv run crossing-smoke --pretrained
```

Expected output shape is `[batch_size, 2]` logits: class `0` is no crossing, class `1` is crossing.

## Train

```powershell
uv run crossing-train `
  --data-root data `
  --output-dir runs/fmow-resnet50 `
  --positive-class crossing `
  --image-size 250 `
  --batch-size 16 `
  --freeze-epochs 12 `
  --finetune-epochs 15
```

Useful options:

- `--backbone fmow-resnet50`: default, recommended true CNN backbone.
- `--backbone sentinel2-resnet18-rgb` or `--backbone sentinel2-resnet50-rgb`: fallback options if the data source changes.
- `--no-pretrained`: debug architecture without downloading pretrained weights.
- `--full-finetune-epochs N`: optional final stage that unfreezes the entire backbone.
- `--pos-weight auto`: default; compensates for class imbalance in `CrossEntropyLoss` by weighting the positive class.
- `--min-finetune-f1 0.70`: start fine-tuning once the frozen MLP reaches this validation F1.
- `--frozen-patience 3`: backup trigger; start fine-tuning when frozen validation F1 stops improving.
- `--finetune-patience 10`: generous early stopping after fine-tuning starts.
- `--lr-scheduler plateau`: default; uses PyTorch `ReduceLROnPlateau` to lower learning rates when validation F1 stalls.
- `--lr-plateau-factor 0.5`: multiply learning rates by this factor on plateau.
- `--lr-plateau-patience 2`: wait this many stale validation epochs before reducing learning rates.
- `--lr-scheduler inverse`: optional older schedule using `1 / (1 + decay * epoch)`.
- `--qat-eval`: train the normal float32 model, but periodically validate a temporary CPU int8 copy after CNN fine-tuning starts.

Each run writes:

- `best.pt`: best validation F1 checkpoint.
- `best_finetuned.pt`: best validation F1 checkpoint after CNN fine-tuning starts.
- `best_quantized.pt`: float32 checkpoint whose temporary int8 copy had the best validation F1, only when `--qat-eval` is enabled.
- `last.pt`: final checkpoint.
- `run_config.json`: resolved training configuration.
- `metrics_history.json`: epoch-by-epoch metrics.
- `quantized_eval_history.json`: int8 validation history, only when `--qat-eval` is enabled.

To make training deployment-aware without paying int8 conversion cost every epoch:

```powershell
uv run crossing-train `
  --data-root data `
  --output-dir runs/fmow-resnet50-qat-eval `
  --positive-class crossing `
  --image-size 250 `
  --batch-size 16 `
  --freeze-epochs 12 `
  --finetune-epochs 15 `
  --qat-eval `
  --qat-calibration-samples 128 `
  --qat-eval-interval 3 `
  --qat-reference-history runs/full-adaptive-gpu/metrics_history.json
```

This is not classic fake-quant QAT. Training still runs in the original float32 model; after fine-tuning starts, the trainer periodically makes a CPU int8 shadow copy, calibrates it, validates it, and saves the best-performing float32 checkpoint as `best_quantized.pt`. By default it checks the first fine-tune epoch, then every 3 fine-tune epochs, then every epoch near the previously observed best window. The current fallback dense window starts at fine-tune stage epoch 7, matching the earlier full run where the best model was found at layer4 epoch 7.

## Int8 Quantization

Export a CPU inference model with post-training static int8 quantization for both the ResNet CNN convolutions and the MLP classifier linears:

```powershell
uv run crossing-quantize-int8 `
  --checkpoint runs/full-adaptive-gpu/best_finetuned.pt `
  --data-root data `
  --output-dir runs/full-adaptive-gpu/int8 `
  --calibration-samples 256 `
  --batch-size 32
```

This keeps training/checkpoints in float32, then creates a deployment-oriented int8 artifact. Quantized PyTorch `Conv2d`/`Linear` inference is CPU-only in this workflow; output logits remain float32. The command writes:

- `model_int8_fx.pt`: pickled FX graph module plus metadata when the local PyTorch backend supports it.
- `model_int8_torchscript.pt`: TorchScript trace when the local PyTorch backend supports saving it.
- `model_int8_state_dict.pt`: quantized state dict fallback for reconstruction/inspection.
- `quantization_summary.json`: backend, calibration settings, quantized module counts, and validation/test metrics.

## Predict Images

Classify any folder of images with a float32 checkpoint:

```powershell
uv run crossing-predict `
  --checkpoint runs/full-adaptive-gpu-qat-eval/best.pt `
  --input-dir path\to\images `
  --output-dir data\predictions\float-best
```

Or use the int8 TorchScript export:

```powershell
uv run crossing-predict `
  --torchscript runs/full-adaptive-gpu-qat-eval/int8-best-quantized/model_int8_torchscript.pt `
  --input-dir path\to\images `
  --output-dir data\predictions\int8-best `
  --device cpu
```

The command writes `predictions.csv` and, by default, copies images into class folders under the output directory:

```text
output-dir/
  predictions.csv
  crossing/
  no_crossing/
```

## Current Final Result

Final combined-dataset run:

- Float32 checkpoint: `runs/final-combined-gpu-qat-eval-continue/best.pt`
- Held-out test F1: `0.9209`
- Held-out test accuracy: `0.9929`
- Held-out test precision: `0.9137`
- Held-out test recall: `0.9283`
- Int8 export: `runs/final-combined-gpu-qat-eval-continue/int8-best-quantized/model_int8_torchscript.pt`
- Int8 held-out test F1: `0.1338`

The float32 model is the final usable model. The int8 export is kept as an optimization artifact, but it currently misses too many crossings on the merged dataset.

## Submission Assets

Small evidence files should be committed to Git. Large generated assets should be uploaded as GitHub Release assets instead of being committed directly.

Prepare the tracked hand-in evidence:

```powershell
uv run python scripts/prepare_submission_assets.py
```

This writes small JSON files and a manifest under `handin/artifacts/`, which is not ignored by Git.

Prepare large release assets locally:

```powershell
uv run python scripts/prepare_submission_assets.py --include-large
```

This writes large files under `submission_assets/`, which is ignored by Git:

- `best.pt`
- `best_quantized.pt`
- `model_int8_torchscript.pt`
- `model_int8_state_dict.pt`
- `hackathon_dataset_imagefolder.zip`, or `hackathon_dataset_imagefolder.zip.partNNN` if the dataset archive is split for upload

Upload those files to a GitHub Release, for example:

```powershell
gh release create hackathon-submission-assets-v1 (Get-ChildItem submission_assets -File).FullName `
  --title "Hackathon submission assets" `
  --notes "Model checkpoints, int8 export, and dataset archive for the cds-108 hackathon submission."
```

The generated `handin/artifacts/submission_asset_manifest.json` records file sizes and SHA-256 checksums so the uploaded assets can be verified. If the dataset archive is split, reassemble the `.partNNN` files in filename order to recover `hackathon_dataset_imagefolder.zip`.

On Windows, after downloading all dataset parts into one folder:

```powershell
cmd /c copy /b hackathon_dataset_imagefolder.zip.part001+hackathon_dataset_imagefolder.zip.part002+hackathon_dataset_imagefolder.zip.part003 hackathon_dataset_imagefolder.zip
```

## AI Assistance

OpenAI Codex was used as a programming and documentation assistant for this project. It helped with implementation planning, PyTorch/TorchGeo integration, dataset import tooling, quantization experiments, debugging, report structure, and wording. The model choices, experiments run, final interpretation, and submission remain the author's responsibility.

## Tests

```powershell
uv run pytest
```

## References

- TorchGeo pretrained weights: https://docs.torchgeo.org/en/stable/api/models.html
- TorchGeo weight loading tutorial: https://docs.torchgeo.org/en/stable/tutorials/pretrained_weights.html
- `timm` feature extraction: https://huggingface.co/docs/timm/v1.0.8/feature_extraction
- SatlasPretrain models: https://github.com/allenai/satlaspretrain_models
