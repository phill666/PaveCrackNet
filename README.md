# PaveCrackNet

**PaveCrackNet: Context-Selective Semantic Guidance and Lightweight Boundary-Aware Supervision for Efficient Pavement Crack Segmentation**

Manuscript being prepared for submission to Pattern Analysis and Applications (PAA).

## Project Overview
PaveCrackNet is a lightweight PyTorch framework for efficient pavement crack segmentation, built on RHACrackNet. It combines cross-layer feature selection with boundary-aware auxiliary supervision to support crack delineation and structural preservation. Performance remains dataset- and seed-dependent.

## Main Contributions
- **Context-Selective Semantic Guidance (CSSG):** shallow and upsampled deep features produce spatial scores that selectively modulate the skip feature: `M_fp = p_s * (1 - p_d)` and `X_out = X_s * (1 - beta * M_fp)`, with learnable `beta`.
- **Lightweight Boundary-Aware Supervision (LBAS):** an auxiliary decoder boundary head encourages boundary preservation during training. The segmentation-only export omits this head.
- **Efficient segmentation:** approximately 0.59M deployment parameters, with the computational cost and measured forward speed reported below.

## Requirements
Use Windows, Python 3.10 and PyTorch 2.0.0+cu118 with CUDA 11.8. From the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Dependencies are listed in [requirements.txt](requirements.txt). Installation in a fresh environment and Linux support have not been verified.

## Dataset Preparation
The study uses three crack segmentation datasets with the following fixed partitions:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| TUT | 987 | 139 | 282 |
| CamCrack789 | 437 | 109 | 243 |
| PY-CrackDB | 258 | 55 | 56 |

See [dataset preparation](docs/DATASETS.md) for directory layouts, preprocessing and partition details. Preserve the original splits. Dataset images and masks are not bundled.

## Training
Edit `dataset_root`, `output_dir` and `device` in the relevant configuration. Set `model_name` to `baseline`, `cssg` or `lbas`; `lbas` selects complete PaveCrackNet training. Use a new output directory for each run.

```powershell
python scripts/train.py --config configs/tut.yaml
python scripts/train.py --config configs/camcrack789.yaml
python scripts/train.py --config configs/pycrackdb.yaml
```

Defaults are seed 2026, 100 epochs, batch size 16 and 512 x 512 inputs. See [training details](docs/TRAINING.md) and [reproducibility](docs/REPRODUCIBILITY.md) for the preserved protocol.

## Evaluation and Inference
Provide an existing dataset and a trusted checkpoint matching the selected model. The paths assigned to `$checkpoint` and `$inputImage` below are placeholders: replace them with your own existing files. No model weights are bundled.

```powershell
$checkpoint = 'path/to/your/existing/training_checkpoint.pth'
python scripts/evaluate.py --dataset tut --data-root data/TUT --model lbas --checkpoint $checkpoint --device cpu
```

For other datasets, use `--dataset camcrack789` or `--dataset pycrackdb` and the corresponding data root. Evaluation uses the checkpoint's frozen Validation threshold; use `--threshold` only to supply a separately frozen Validation value, never a Test-tuned threshold.

Export the segmentation-only model and use `--model cssg` to load it:

```powershell
python scripts/export_model.py --checkpoint $checkpoint --output checkpoints/deployment.pth
$inputImage = 'path/to/your/existing/image.png'
python scripts/predict.py --input $inputImage --output outputs/predict --model cssg --checkpoint checkpoints/deployment.pth --device cpu
```

The export command creates the output checkpoint; it is not a pre-existing downloadable artifact. Prediction also accepts an existing image folder. See [evaluation and inference details](docs/EVALUATION.md). Model outputs are probabilities, so no additional sigmoid is required.

## Experimental Results
The manuscript reports the following PaveCrackNet F1 scores:

| Dataset | F1 (%) |
| --- | ---: |
| TUT | 81.32 |
| CamCrack789 | 81.16 |
| PY-CrackDB | 58.56 |

These reported values are retained without changing historical experimental records. See [reproducibility scope](docs/REPRODUCIBILITY.md) for record and checkpoint limitations.

## Computational Efficiency
| Deployment parameters | Counted FLOPs | Pure network FPS |
| ---: | ---: | ---: |
| Approximately 0.59M | 8.20G | 105.06 |

Measured on an **RTX 4090**, with **512 x 512** inputs, **batch size 1**, **FP32**, **PyTorch 2.0.0+cu118** and **CUDA 11.8**. FPS refers to pure network forward execution, excluding data reading, preprocessing and external postprocessing; it is not end-to-end road-inspection system throughput. FLOP counting uses 2 FLOPs per MAC and does not cover every runtime operator.

## Code Availability
Model source, configurations and usage tools are provided in this repository. Dataset images and model weights are not bundled. Consult [dataset preparation](docs/DATASETS.md) and [reproducibility](docs/REPRODUCIBILITY.md) for the available metadata and requirements for exact reproduction.

The baseline derives from [RHACrackNet](https://github.com/lovelyyoshino/RHACrackNet), accompanying *A lightweight encoder-decoder network for automatic pavement crack detection* ([paper](https://doi.org/10.1111/mice.13103)). Original attribution is retained.

## Citation
The following manuscript is being prepared for PAA and is not yet published:

> PaveCrackNet: Context-Selective Semantic Guidance and Lightweight Boundary-Aware Supervision for Efficient Pavement Crack Segmentation

A finalized bibliographic entry is not available. No acceptance, publication date or DOI is claimed for PaveCrackNet.

## License
Inherited RHACrackNet portions retain their [MIT notice](third_party/RHACrackNet_LICENSE). Licensing of authorized original contributions awaits final copyright-holder confirmation, and DyReLU remains **`LICENSE_UNRESOLVED`**. See [LICENSE_STATUS.md](LICENSE_STATUS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the component-specific scope and outstanding permissions.
