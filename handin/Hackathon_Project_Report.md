# Hackathon Project Report

This project builds a PyTorch classifier for detecting pedestrian crossings in high-resolution Swiss aerial imagery. The task is binary classification: each `25 m x 25 m` RGB patch is classified as either `crossing` or `no_crossing`.

At the target resolution of `10 cm/pixel`, each patch is loaded as a `250 x 250` RGB image. The final model uses a remote-sensing-pretrained ResNet-50 CNN as feature extractor and a custom MLP classification head. The best final model is the float32 checkpoint, which reached a held-out test F1 score of `0.9209` on the merged dataset.

## Dataset

The dataset uses a simple two-class `ImageFolder` layout:

```text
data/
  train/
    crossing/
    no_crossing/
  val/
    crossing/
    no_crossing/
  test/
    crossing/
    no_crossing/
```

The class mapping is fixed in the training code: `crossing` is positive label `1`, and `no_crossing` is negative label `0`.

The first dataset was reconstructed from public dataset metadata released by Oliver Schütz (`DotNaos`) with permission to reuse the dataset part only. No model code from that repository was used. The source repository was:

```text
https://github.com/DotNaos/fs26-crosswalk-detector
```

This base dataset is balanced:

| Split | crossing | no_crossing | Total |
| --- | ---: | ---: | ---: |
| Train | 2000 | 2000 | 4000 |
| Validation | 250 | 250 | 500 |
| Test | 250 | 250 | 500 |
| Total | 2500 | 2500 | 5000 |

I then added my own dataset from `data/svenzebradata.zip`. Its original structure used `zebra/y` for crossing images and `zebra/n` for non-crossing images. I imported it into the same `ImageFolder` layout with a deterministic `80/10/10` split.

The additional dataset is strongly imbalanced:

| Split | crossing | no_crossing | Total |
| --- | ---: | ---: | ---: |
| Train | 14 | 40946 | 40960 |
| Validation | 2 | 5118 | 5120 |
| Test | 1 | 5119 | 5120 |
| Total | 17 | 51183 | 51200 |

After merging both sources, the final local dataset contains:

| Split | crossing | no_crossing | Total |
| --- | ---: | ---: | ---: |
| Train | 2014 | 42946 | 44960 |
| Validation | 252 | 5368 | 5620 |
| Test | 251 | 5369 | 5620 |
| Total | 2517 | 53683 | 56200 |

This final dataset is therefore heavily skewed toward `no_crossing`. Accuracy alone would be misleading, so the final evaluation focuses on F1, precision, recall, and the confusion matrix. During training, I used automatic class weighting so that the minority positive class still had a meaningful influence on the loss.

The imagery is well suited for the task because pedestrian crossings are visible road markings in high-resolution aerial images. The main limitations are possible label noise, the strong class imbalance after importing my own data, and the fact that the split is stratified but not guaranteed to be geographically separated. Nearby SwissImage patches can look very similar, so a future version should prefer a more explicit geographic split.

## Model Architecture

The model is a CNN-based transfer-learning classifier:

- Backbone: ResNet-50
- Implementation: `timm`
- Pretrained weights: TorchGeo `ResNet50_Weights.FMOW_RGB_GASSL`
- Domain of pretraining: overhead / remote-sensing RGB imagery
- Framework: PyTorch

I chose this backbone because the input images are aerial images, so remote-sensing pretraining is a better fit than a generic ImageNet-only initialization. ResNet-50 is also a strict CNN, which matched the model direction for this project.

The original ResNet classifier is removed. The ResNet backbone outputs a `2048`-dimensional feature vector after global average pooling, and this vector is passed into a custom MLP:

```text
2048 -> 2048 -> 256 -> 32 -> 2
```

The model output is a two-logit tensor:

```text
[B, 2]
```

Class index `0` corresponds to `no_crossing`, and class index `1` corresponds to `crossing`.

Parameter counts:

| Component | Parameters |
| --- | ---: |
| Full model | 28,237,218 |
| ResNet backbone | 23,508,032 |
| Custom MLP head | 4,729,186 |
| Fine-tuned ResNet `layer4` block | 14,964,736 |

Most parameters come from the pretrained CNN. The custom head is intentionally fairly large, because the CNN provides reusable visual features while the MLP learns the final crossing/no-crossing decision boundary.

## Training And Optimization

Training is staged rather than fine-tuning the whole network immediately.

First, the CNN backbone is frozen and only the MLP head is trained. This avoids pushing random gradients from an untrained classifier head through the pretrained CNN too early. Once the head reaches a reasonable validation F1, or once patience-based backup criteria trigger, the last ResNet stage (`layer4`) is unfrozen and fine-tuned.

The final run used:

- optimizer: AdamW
- head learning rate: `1e-3` initially
- backbone learning rate: `1e-4` initially
- scheduler: PyTorch `ReduceLROnPlateau`
- scheduler monitor: validation F1
- dropout in the MLP head: `0.2`
- weight decay: `1e-4`
- class weighting: automatic positive-class weighting
- fine-tuning early stopping: enabled

The training augmentations were simple but useful for aerial imagery:

- random horizontal flip
- random vertical flip
- random rotation up to 180 degrees
- color jitter

The final training was run locally on an NVIDIA GeForce RTX 3070 Ti. I know that the course setup expected use of the provided servers, but the VPN workflow became a practical obstacle. Since I had a capable local GPU available, I trained the final model locally.

I also implemented post-training int8 quantization for both the CNN convolutions and the MLP linear layers. The exported int8 model uses CPU inference with the `onednn` backend and is saved as TorchScript. In addition, I implemented a QAT-style evaluation loop: the model still trains in float32, but after fine-tuning starts, temporary int8 copies are periodically created and validated. This is not classic fake-quantization QAT; it is better described as periodic int8 shadow evaluation during float32 training.

## Results

The final training used the merged classmate and own dataset. The first long run reached epoch `24`; validation F1 was still improving, so I continued from `last.pt` with lower learning rates. The continuation run stopped after `14` more epochs due to fine-tuning early stopping. The best checkpoint was found at continuation epoch `6`.

Best float32 checkpoint:

```text
runs/final-combined-gpu-qat-eval-continue/best.pt
```

Validation metrics for the best float32 checkpoint:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9923 |
| Precision | 0.8885 |
| Recall | 0.9484 |
| F1 | 0.9175 |

Held-out test metrics for the same checkpoint:

| Metric | Value |
| --- | ---: |
| Loss | 0.0258 |
| Accuracy | 0.9929 |
| Precision | 0.9137 |
| Recall | 0.9283 |
| F1 | 0.9209 |
| TP | 233 |
| FP | 22 |
| TN | 5347 |
| FN | 18 |

The float32 model is the final usable model. Despite the strong class imbalance, it keeps both precision and recall high on the held-out test split.

The int8 export was much less successful:

```text
runs/final-combined-gpu-qat-eval-continue/int8-best-quantized/model_int8_torchscript.pt
```

Held-out test metrics for the int8 export:

| Metric | Value |
| --- | ---: |
| Loss | 0.1137 |
| Accuracy | 0.9585 |
| Precision | 1.0000 |
| Recall | 0.0717 |
| F1 | 0.1338 |
| TP | 18 |
| FP | 0 |
| TN | 5369 |
| FN | 233 |

The int8 model almost never predicts `crossing`, which gives perfect precision but unusably low recall. I therefore treat quantization as an implemented but unsuccessful deployment optimization for the final merged dataset. A more robust version would need proper fake-quantization QAT or a better balanced calibration set.

All tests passed after the final implementation:

```text
15 passed
```

## Reflection

This was my first time implementing a fine-tuning workflow end to end. The main thing I learned is that fine-tuning should not necessarily start immediately. If the classifier head is still too random, unfreezing the CNN too early can make the model worse or less stable. The staged training setup made that behavior easier to control.

The most difficult part was quantizing the pretrained CNN, not the custom MLP. Quantizing only the classifier head would have been much simpler, but the goal was to quantize both the ResNet backbone and the MLP head. This required a separate inference path that could be traced and quantized cleanly.

It also surprised me how sensitive the quantized model was on the final merged dataset. Earlier quantization experiments looked much more promising, but after adding the large imbalanced own dataset, the int8 model collapsed toward predicting almost everything as `no_crossing`. That result is still useful: it shows that the float32 model is strong, but the deployment version needs more work.

If I continued this project, I would implement proper fake-quantization QAT instead of the current slow shadow-evaluation method. I would also revisit the dataset balance, especially the calibration set for quantization, because the final dataset contains many more negative samples than positive samples.

## Disclosure

Tools and external resources used:

- PyTorch, torchvision, timm, TorchGeo, tqdm, Pillow
- `uv` / Astral for package and environment management
- CUDA GPU training on NVIDIA GeForce RTX 3070 Ti
- TorchGeo pretrained weights: `ResNet50_Weights.FMOW_RGB_GASSL`
- Dataset metadata/reconstruction source: Oliver Schütz (`DotNaos`), `https://github.com/DotNaos/fs26-crosswalk-detector`, used with permission for the dataset part only

AI assistance was used as a programming and documentation aid. It helped with implementation planning, code scaffolding, debugging, PyTorch/TorchGeo integration, quantization experiments, report structure, and wording. The model choices, experiments run, final interpretation, and submission remain my responsibility.
