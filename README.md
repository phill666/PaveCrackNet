# PaveCrackNet

**PaveCrackNet: Context-Selective Semantic Guidance and Lightweight Boundary-Aware Supervision for Real-Time Pavement Crack Segmentation**

Manuscript being prepared for submission to the Journal of Real-Time Image Processing (JRTIP).

## Introduction
PaveCrackNet is a lightweight PyTorch crack segmentation model built on RHACrackNet*. It combines context-selective skip-feature modulation with training-time boundary supervision. Performance varies across datasets and random seeds; universal superiority is not claimed.

## Main Contributions
- **CSSG:** shallow and upsampled deep features produce spatial scores. The gate uses `M_fp = p_s * (1 - p_d)` and `X_out = X_s * (1 - beta * M_fp)`, with learnable `beta`.
- **LBAS:** a lightweight auxiliary boundary head supervises decoder features during training. Deployment export retains only the segmentation network.
- **Lightweight segmentation:** approximately 0.59M deployment parameters with the reported efficiency below.

## Requirements
Use Python 3.10 and PyTorch 2.0.0. The reference CUDA environment is CUDA 11.8. From the repository root in Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for Python dependencies. Linux support and installation in a fresh environment have not been verified.

## Dataset Preparation
Obtain images and masks from their authorized providers; none are bundled. Preserve the original fixed partitions:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| TUT | 987 | 139 | 282 |
| CamCrack789 | 437 | 109 | 243 |
| PY-CrackDB | 258 | 55 | 56 |

TUT uses fixed folders and includes multiple structural materials. PY-CrackDB split lists are in [splits/pycrackdb](splits/pycrackdb). CamCrack789 uses a seed-2026 split of the provided 546 training images; the 243 provided validation images form Test. The original Cam lists are verified locally but remain unbundled pending redistribution permission. Do not generate a replacement split.

Follow [dataset preparation and original split hashes](docs/DATASETS.md). After obtaining the original Cam lists:

```powershell
python scripts/validate_cam_splits.py --data-root data/CamCrack789 --check-image-content
```

## Training
Edit `dataset_root`, `output_dir` and `device` in the relevant YAML file. Choose `model_name: baseline`, `cssg` or `lbas`; `lbas` is the complete PaveCrackNet training model. Use a new output directory for each run.

```powershell
python scripts/train.py --config configs/tut.yaml
python scripts/train.py --config configs/camcrack789.yaml
python scripts/train.py --config configs/pycrackdb.yaml
```

The default protocol uses seed 2026, 100 epochs, batch size 16 and 512 x 512 inputs. See [training details](docs/TRAINING.md) and [reproducibility](docs/REPRODUCIBILITY.md) for losses, optimizer, checkpoint selection and protocol limitations.

## Evaluation
Supply a trusted checkpoint matching the selected model and an existing dataset. For a complete-model training checkpoint:

```powershell
python scripts/evaluate.py --dataset tut --data-root data/TUT --model lbas --checkpoint checkpoints/best_model.pth --device cpu
```

Use `--dataset camcrack789` or `--dataset pycrackdb` with the corresponding data root. Evaluation uses the checkpoint's frozen Validation threshold; if a separate Validation calibration was used, pass that frozen value through `--threshold`. Never choose a threshold on Test.

Export segmentation-only weights:

```powershell
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --output checkpoints/deployment.pth
```

Load this export with `--model cssg`. For prediction, supply your own existing input image or folder:

```powershell
python scripts/predict.py --input path/to/image.png --output outputs/predict --model cssg --checkpoint checkpoints/deployment.pth --device cpu
```

See [evaluation and prediction](docs/EVALUATION.md) for Validation calibration and output details. Networks output probabilities; no extra sigmoid is required.

## Experimental Results
The current manuscript reports the following PaveCrackNet results:

| Dataset | F1 (%) |
| --- | ---: |
| TUT | 81.32 |
| CamCrack789 | 81.16 |
| PY-CrackDB | 58.56 |

These manuscript-reported values are preserved without rerunning experiments or changing historical records. The latest manuscript-to-checkpoint mapping is not independently verified in this release; see [reproducibility scope](docs/REPRODUCIBILITY.md).

## Model Efficiency
| Deployment parameters | Counted FLOPs | Pure network FPS |
| ---: | ---: | ---: |
| Approximately 0.59M | 8.20G | 105.06 |

Reported on an **RTX 4090**, **512 x 512** input, **batch size 1**, **FP32**, PyTorch **2.0.0+cu118** / CUDA **11.8**. FPS measures pure network forward execution, not end-to-end preprocessing, I/O or on-vehicle latency. FLOP counting uses 2 FLOPs per MAC and does not cover every runtime operator.

## Code Availability
This repository provides model source, training configurations, evaluation/prediction tools and PY-CrackDB split metadata. Model weights and datasets are not bundled. Exact reproduction also requires the original dataset partitions and matching checkpoints; see [reproducibility scope](docs/REPRODUCIBILITY.md).

The inherited baseline is based on [RHACrackNet](https://github.com/lovelyyoshino/RHACrackNet), accompanying *A lightweight encoder-decoder network for automatic pavement crack detection* ([paper](https://doi.org/10.1111/mice.13103)). Original notices are retained.

## Citation
The PaveCrackNet manuscript is unpublished. Refer to the full title above; a finalized bibliographic entry is not yet available. No acceptance, publication date or DOI is claimed for PaveCrackNet.

## License
Inherited RHACrackNet portions retain their [MIT notice](third_party/RHACrackNet_LICENSE). MIT licensing is intended only for authorized original contributions, with the final copyright-holder identity still pending.

**DyReLU remains `LICENSE_UNRESOLVED`.** No explicit redistribution permission has been verified; the repository must not be treated as a fully permission-cleared release. See [license status and next steps](LICENSE_STATUS.md) and [third-party notices](THIRD_PARTY_NOTICES.md) for the component-specific scope.
