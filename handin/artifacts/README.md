# Hand-In Artifacts

This folder contains small, tracked evidence files for the hackathon report.

Large binary assets are intentionally not committed to Git:

- float32 model checkpoints (`best.pt`, `best_quantized.pt`)
- int8 TorchScript/state-dict exports
- dataset image archive, split into `.partNNN` files when it is large

Use `uv run python scripts/prepare_submission_assets.py --include-large` to
prepare those large files under `submission_assets/`, then upload them as GitHub
Release assets. The generated `submission_asset_manifest.json` contains file
sizes and SHA-256 checksums so the release assets can be verified. If the
dataset archive was split, reassemble the `.partNNN` files in filename order to
recover `hackathon_dataset_imagefolder.zip`.

Windows example after downloading all parts into one folder:

```powershell
cmd /c copy /b hackathon_dataset_imagefolder.zip.part001+hackathon_dataset_imagefolder.zip.part002+hackathon_dataset_imagefolder.zip.part003 hackathon_dataset_imagefolder.zip
```
