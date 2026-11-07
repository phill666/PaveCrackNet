# Training
From the repository root in Windows PowerShell:
```powershell
python scripts/train.py --config configs/tut.yaml
python scripts/train.py --config configs/camcrack789.yaml
python scripts/train.py --config configs/pycrackdb.yaml
```
Edit dataset_root/output_dir/device and model_name (baseline, cssg, lbas) for nine runs; outputs must be new directories. Nominal batch16, epochs100, LR.001, seed2026, input512 and interval10 are deliberately locked. Other values require the explicit --allow-custom flag; custom runs are not paper reproduction. learning_rate is applied to the optimizer only for custom runs. Dataset/model/output/device and other YAML keys are consumed by the dispatcher. CPU is selectable but historical training code retains CUDA seed calls and has not been dynamically validated on CPU.
TUT baseline/CSSG entry functions are main(), not train(). Historical TUT/Cam/PY training loop semantics, losses, augmentation and tie rules remain intact. Smoke-only helpers and modes are excluded; use documented commands. No reproduction has been run for this release.
