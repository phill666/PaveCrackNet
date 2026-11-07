# Evaluation and prediction
No checkpoints are included. Use trusted matching checkpoints: baseline, cssg (also the deployment export), or lbas (training wrapper). Strict state loading rejects mismatched prefixes and missing keys. Networks already output probabilities: do not add sigmoid.
```powershell
python scripts/evaluate.py --dataset tut --data-root data/TUT --model lbas --checkpoint checkpoints/best_model.pth --device cpu
python scripts/predict.py --input examples --output outputs/predict --model lbas --checkpoint checkpoints/best_model.pth --device cpu
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --output checkpoints/deployment.pth
```
Input can be an image or folder. Prediction masks/overlays are 512x512, not original resolution. --threshold explicitly overrides the checkpoint frozen Val threshold; never tune it on Test. CUDA unavailable raises an error; choose --device cpu. Export writes only the segmentation state; load it as --model cssg. This lightweight export wrapper does not repeat the original deployment equivalence forward test. All new wrappers passed static checks only; their numerical equivalence has not been dynamically verified.

For fixed-checkpoint V2 validation calibration add --split val --calibrate-val. The search is .05:.01:.99 and reports a threshold; it does not overwrite the checkpoint. Use that frozen threshold explicitly for subsequent Test. Historical checkpoint selection remains separate.

Single-image example (supply your own existing image and trusted weights):
```powershell
python scripts/predict.py --input path/to/image.png --output outputs/single --model lbas --checkpoint checkpoints/best_model.pth --device cpu
```
Folder input uses the same command with --input path/to/images. Example weight paths are user-supplied prerequisites, not bundled/downloadable artifacts.
