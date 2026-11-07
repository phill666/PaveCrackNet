# PaveCrackNet

Context-Selective Semantic Guidance and Lightweight Boundary-Aware Supervision for Real-Time Pavement Crack Segmentation

PyTorch source preparation for the manuscript **PaveCrackNet: Context-Selective Semantic Guidance and Lightweight Boundary-Aware Supervision for Real-Time Pavement Crack Segmentation** (target journal IJPE; publication/acceptance not claimed).

## Introduction and highlights
A lightweight pavement crack segmenter with CSSG spatial feature modulation and LBAS auxiliary boundary supervision. Deployment exports omit the auxiliary boundary head. Performance is dataset-dependent, with limitations on PY-CrackDB; no claim of universal best performance is made.

## Architecture
RHACrackNet* backbone; E2 skip is modulated using shallow/deep 1x1 sigmoid scores: M=ps*(1-pd), Xout=Xs*(1-beta*M), beta initially .1. D3 supervises a depthwise boundary head, with logits upsampled before loss. Final manuscript figure identities/redistribution permissions were not established, so figures are intentionally not bundled or redrawn.

## Installation (Windows PowerShell)
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```
Use Python3.10 for the historical torch build. CUDA toolkit/library compatibility is the user's responsibility. Installation was not executed for this preparation; Linux is unverified.

## Dataset preparation
See [DATASETS](docs/DATASETS.md). No data or weights are supplied. Missing permission-cleared Cam split lists limit independent exact reproduction.

## Training
See [TRAINING](docs/TRAINING.md); edit configs/*.yaml, then `python scripts/train.py --config configs/tut.yaml`. Three dataset configurations x baseline/cssg/lbas map to nine preserved formal entry paths.

## Evaluation and inference
See [EVALUATION](docs/EVALUATION.md). Checkpoints are required and not bundled. Single-image/folder prediction, threshold override, CPU/CUDA, overlays and quantitative evaluation are provided as statically checked wrappers only.

## Experimental results
The reported final PaveCrackNet F1 values are TUT **80.91%**, CamCrack789 **81.16%**, PY-CrackDB **56.48%** (historical locked records, not rerun for this release). LBAS does not consistently improve PY-CrackDB relative to CSSG-only.

## Model complexity
Deployment approximately **0.59M parameters**, **8.20G counted FLOPs**, **105.06 FPS**. FPS is pure network forward speed on RTX4090, FP32, batch1, 512square; not end-to-end on-vehicle latency. MAC counting uses 2FLOPs/MAC and does not cover every runtime operator.

## Reproducibility
See [REPRODUCIBILITY](docs/REPRODUCIBILITY.md). No training, testing, inference or benchmarking was run during packaging. Static success does not establish full reproduction.

## Citation
The manuscript is unpublished; cite the title above if referring to this work. Author metadata, DOI, volume, pages and acceptance status are intentionally not fabricated. A finalized bibliographic entry is not supplied.

## Acknowledgements
The inherited RHACrackNet baseline is based on the implementation accompanying *A lightweight encoder-decoder network for automatic pavement crack detection*, DOI https://doi.org/10.1111/mice.13103 . Original copyright/license notices are retained. External comparison model sources are not redistributed.

## License
MIT has been selected for authorized PaveCrackNet original contributions; the final copyright-holder line is pending confirmation. Inherited RHACrackNet portions retain their original MIT license. Other components retain their respective terms; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) and [LICENSE_STATUS](LICENSE_STATUS.md).
This local release candidate still contains a required DyReLU implementation with unresolved upstream permission. It must not be publicly distributed until that permission is resolved. No assertion that all bundled code is authorized is made. No public repository has been created by this preparation.
