#!/usr/bin/env python3
"""Create balanced Jacquard V2 sample lists using YCB-style object groups.

Requires the extracted Jacquard V2 dataset and ShapeNetSem v0 metadata.csv.
No image or annotation files are moved or copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


GROUPS = ("food", "kitchen", "tools", "tasks")
CSV_FIELDS = ("sample_path", "grasp_path", "object_id", "categories", "group")


def normalize_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def model_id(value: str) -> str:
    """ShapeNetSem fullId is usually `wss.<32-character model id>`."""
    value = value.strip()
    if value.startswith("wss."):
        value = value[4:]
    return value.casefold()


def read_mapping(path: Path) -> dict[str, set[str]]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if set(raw) != set(GROUPS):
        raise ValueError(f"Mapping must contain exactly these groups: {', '.join(GROUPS)}")
    mapping = {group: {normalize_category(x) for x in raw[group]} for group in GROUPS}
    duplicated = defaultdict(list)
    for group, categories in mapping.items():
        for category in categories:
            duplicated[category].append(group)
    overlaps = {key: groups for key, groups in duplicated.items() if len(groups) > 1}
    if overlaps:
        raise ValueError(f"Categories mapped to more than one group: {overlaps}")
    return mapping


def read_metadata(path: Path) -> dict[str, tuple[str, ...]]:
    objects: dict[str, tuple[str, ...]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"fullId", "category"} <= set(reader.fieldnames):
            raise ValueError("ShapeNetSem metadata must have fullId and category columns")
        for row in reader:
            object_id = model_id(row["fullId"] or "")
            if not object_id:
                continue
            categories = tuple(c.strip() for c in (row["category"] or "").split(",") if c.strip())
            if object_id in objects and objects[object_id] != categories:
                raise ValueError(f"Conflicting category metadata for {object_id}")
            objects[object_id] = categories
    return objects


def discover_samples(dataset: Path) -> dict[str, list[tuple[Path, Path]]]:
    objects: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for grasp_file in sorted(dataset.rglob("*_grasps.txt")):
        # Each scene has one grasp file and one RGB image with the same prefix.
        rgb_file = grasp_file.with_name(grasp_file.name.removesuffix("_grasps.txt") + "_RGB.png")
        if not rgb_file.is_file():
            raise FileNotFoundError(f"Missing RGB image for {grasp_file}: {rgb_file}")
        object_id = model_id(grasp_file.parent.name)
        if not object_id:
            raise ValueError(f"Cannot identify object folder for {grasp_file}")
        objects[object_id].append((rgb_file.relative_to(dataset), grasp_file.relative_to(dataset)))
    if not objects:
        raise ValueError(f"No *_grasps.txt files found under {dataset}")
    return objects


def balance_objects(
    objects: dict[str, list[tuple[Path, Path]]], target: int, seed: int
) -> list[str]:
    ids = sorted(objects)
    random.Random(seed).shuffle(ids)
    chosen: list[str] = []
    remaining = target
    for object_id in ids:
        count = len(objects[object_id])
        if count <= remaining:
            chosen.append(object_id)
            remaining -= count
        if remaining == 0:
            break
    return chosen


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted Jacquard V2 root")
    parser.add_argument("--metadata", type=Path, required=True, help="ShapeNetSem v0 metadata.csv")
    parser.add_argument("--output", type=Path, required=True, help="Directory for five CSV files")
    parser.add_argument("--mapping", type=Path, default=Path(__file__).with_name("ycb_category_mapping.json"))
    parser.add_argument("--seed", type=int, default=0, help="Reproducible object selection")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve(strict=True)
    metadata = args.metadata.expanduser().resolve(strict=True)
    mapping = read_mapping(args.mapping)
    categories_by_object = read_metadata(metadata)
    samples_by_object = discover_samples(dataset)

    grouped: dict[str, dict[str, list[tuple[Path, Path]]]] = {group: {} for group in GROUPS}
    skipped = Counter()
    for object_id, samples in samples_by_object.items():
        categories = categories_by_object.get(object_id)
        if categories is None:
            skipped["no_metadata"] += len(samples)
            continue
        hits = [group for group in GROUPS if any(normalize_category(c) in mapping[group] for c in categories)]
        if len(hits) == 1:
            grouped[hits[0]][object_id] = samples
        else:
            skipped["ambiguous" if hits else "unmapped"] += len(samples)

    totals = {group: sum(map(len, grouped[group].values())) for group in GROUPS}
    empty = [group for group, count in totals.items() if not count]
    if empty:
        raise ValueError(
            f"No eligible samples for {', '.join(empty)}. Inspect the mapping and dataset; "
            "no CSVs were written."
        )
    target = min(totals.values())
    args.output.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        selected = balance_objects(grouped[group], target, args.seed)
        rows = [
            {
                "sample_path": rgb.as_posix(),
                "grasp_path": grasp.as_posix(),
                "object_id": object_id,
                "categories": ",".join(categories_by_object[object_id]),
                "group": group,
            }
            for object_id in selected
            for rgb, grasp in grouped[group][object_id]
        ]
        rows.sort(key=lambda row: row["sample_path"])
        write_csv(args.output / f"{group}.csv", rows)
        print(f"{group:8} {len(rows):6} samples  {len(selected):5} objects  ({totals[group]} eligible)")
    print(f"Excluded samples: {dict(skipped)}")
    print("Paths in the CSVs are relative to --dataset.")


if __name__ == "__main__":
    main()
