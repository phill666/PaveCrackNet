"""Validate original CamCrack789 split files; never generate or change a split."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath

EXPECTED = {
    "train_seed2026.txt": (437, "a5a4aa85e2454db59facf6ca8c7993ff262e280a553be6bf9499d5d1189eaf5b", "train"),
    "val_seed2026.txt": (109, "bdf3e1fd568386d76597e61393e9ab1c27e2c2b951cd9864a9f1f59aa8f8f83a", "train"),
    "test_provided_valid.txt": (243, "367438a4f8407ca4135b4c82dbe74923300140020e1637b292cc8490223c8af5", "valid"),
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def relative_file(root, value):
    # Reject Windows drive/UNC paths even when this validator runs on Linux.
    value = value.replace(chr(92), "/")
    rel = PurePosixPath(value)
    if not value or rel.is_absolute() or ":" in value or ".." in rel.parts:
        raise ValueError("Expected a safe dataset-relative path")
    path = (root / Path(*rel.parts)).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Missing file or path outside dataset root: " + value)
    return path

def validate(data_root, split_dir=None, check_image_content=False):
    root = Path(data_root).resolve()
    folder = Path(split_dir).resolve() if split_dir else root / "splits"
    pairs, checks = {}, {}
    for name, (count, expected_hash, source) in EXPECTED.items():
        file = folder / name
        actual_hash = sha256(file)
        if actual_hash != expected_hash:
            raise ValueError(name + ": SHA256 does not match the original experiment")
        entries = []
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) != 2:
                raise ValueError(name + ": expected relative_image|relative_mask")
            image, mask = [relative_file(root, field) for field in fields]
            if image.parent != root / source / "image" or mask.parent != root / source / "seg_gt":
                raise ValueError(name + ": wrong original train/valid source")
            if not image.stem.startswith("image-") or not mask.stem.startswith("target-") or image.stem[6:] != mask.stem[7:]:
                raise ValueError(name + ": image/mask ID mismatch")
            entries.append((image, mask))
        if len(entries) != count or any(len({p[j] for p in entries}) != count for j in (0, 1)):
            raise ValueError(name + ": wrong count or duplicate image/mask path")
        pairs[name] = entries
        checks[name] = {"count": count, "sha256": actual_hash}
    for a, b in itertools.combinations(pairs, 2):
        for column in (0, 1):
            if {p[column] for p in pairs[a]} & {p[column] for p in pairs[b]}:
                raise ValueError("Cross-split path overlap: " + a + " / " + b)
        if check_image_content and {sha256(p[0]) for p in pairs[a]} & {sha256(p[0]) for p in pairs[b]}:
            raise ValueError("Cross-split byte-identical images: " + a + " / " + b)
    train_images = {p[0] for name in ("train_seed2026.txt", "val_seed2026.txt") for p in pairs[name]}
    test_images = {p[0] for p in pairs["test_provided_valid.txt"]}
    for source, actual, count in (("train", train_images, 546), ("valid", test_images, 243)):
        provided = {p.resolve() for p in (root / source / "image").iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")}
        if len(provided) != count or provided != actual:
            raise ValueError("Split coverage differs from provided " + source)
    return {"status": "PASS", "seed": 2026, "files": checks, "path_overlap": False,
            "cross_split_byte_identical_images": False if check_image_content else "NOT_CHECKED",
            "near_duplicate_or_acquisition_group_leakage": "NOT_ESTABLISHED",
            "redistribution_permission": "NOT_ESTABLISHED", "data_or_splits_modified": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir")
    parser.add_argument("--check-image-content", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root, args.split_dir, args.check_image_content), indent=2))

if __name__ == "__main__":
    main()
