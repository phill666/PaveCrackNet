# Dataset preparation
No images, labels, archives or weights are distributed. Do not re-split.

* TUT: retain the provided fixed folders train_img/train_lab, val_img/val_lab, test_img/test_lab (987/139/282). An independently certified official split and download URL are not established in the release evidence. Obtain the data from its rights holder; do not assume a similarly named dataset is identical.
* CamCrack789: custom seed-2026 437/109 split from the provided 546 training images; the 243 provided validation images serve as Test. Place the original fixed train_seed2026.txt, val_seed2026.txt, test_provided_valid.txt in dataset_root/splits. Each line is relative_image_path|relative_mask_path. The exact original lists were located and verified against the historical split manifest and locked experiment records. Metadata redistribution permission is unconfirmed, so these lists are not bundled. Obtain the exact original lists from an authorized source; do not regenerate a random split.
* PY-CrackDB: https://doi.org/10.5281/zenodo.16749554 ; local download record identifies CC-BY-4.0. Attribute PY-CrackDB, record 16749554, version as supplied at the DOI. Seed-2026 grouped/stratified split 258/55/56, 369 annotated pairs / 304 unique original images. Copy bundled splits/pycrackdb/*.txt to dataset_root/splits. Lines are tab-separated relative image and mask paths.

RGB images: bilinear resize to 512 square and float32 /255. TUT masks: grayscale, nearest resize, >0. Cam/PY: PIL convert('1'), nearest resize, binary float32. Training transforms are synchronized H/V/180-degree flips, each p=.5, and image-only brightness x1.5, brightness x.5, contrast x1.5, each independently p=.5. Validation/Test transforms are disabled. TUT content-level leakage has not been fully excluded. Exact-image grouping does not establish independence of every unrecognized same-scene acquisition.

## CamCrack789 original-split identity and validation
The fixed split is custom, not an official three-way split. Train/Val together exhaust the provided 546-image training pool; Test exhausts the provided 243-image validation pool. Counts and ordered list bytes must match the originals:

|File|Pairs|SHA256 (original file bytes)|
|---|---:|---|
|train_seed2026.txt|437|a5a4aa85e2454db59facf6ca8c7993ff262e280a553be6bf9499d5d1189eaf5b|
|val_seed2026.txt|109|bdf3e1fd568386d76597e61393e9ab1c27e2c2b951cd9864a9f1f59aa8f8f83a|
|test_provided_valid.txt|243|367438a4f8407ca4135b4c82dbe74923300140020e1637b292cc8490223c8af5|

After obtaining the original lists and dataset through an authorized source, run:
```powershell
python scripts/validate_cam_splits.py --data-root data/CamCrack789 --check-image-content
```
Use `--split-dir path/to/original/splits` if the lists are elsewhere. The read-only validator checks original SHA256, counts, relative paths, image/mask IDs, train/valid pool coverage, path overlaps and optional cross-split byte-identical images. It never generates a split or grants redistribution rights. A hash mismatch fails even if only line endings changed. Byte-content checks do not exclude all near duplicates or related acquisitions. The loader reads these same original lists with no random repartitioning.

Dataset provenance: the [RHACrackNet upstream project](https://github.com/lovelyyoshino/RHACrackNet) describes CamCrack789, but its software MIT notice has not been independently established as permission to redistribute this dataset or these derived split lists. No images, masks or unapproved lists are added here.
