# YCB-style Jacquard V2 splits

`split_jacquard_v2_ycb.py` joins Jacquard V2 object folder IDs to the
`fullId` and `category` columns of ShapeNetSem v0 `metadata.csv`. It maps
ShapeNetSem categories into five *approximate* YCB groups using the editable
`ycb_category_mapping.json` file. The mapping is a research choice, not an
official equivalence between the two datasets. In particular, `Bottle` is
treated as food packaging and `Rock` as a shape item.

Run with Python 3.9 or newer:

```bash
python split_jacquard_v2_ycb.py \
  --dataset /path/to/JacquardV2 \
  --metadata /path/to/ShapeNetSem.v0/metadata.csv \
  --output /path/to/splits \
  --seed 0
```

The dataset root can contain the four `JacquardV2_Dataset_*` parts or their
extracted object folders. The script finds every `*_grasps.txt` and its
matching `_RGB.png`. It writes `food.csv`, `kitchen.csv`, `tools.csv`,
`shapes.csv`, and `tasks.csv`. Each row contains the RGB sample path, grasp
annotation path, model ID, source categories, and assigned group. Paths are
relative to `--dataset`.

Balancing uses the group with the fewest eligible images as the target. It
selects complete objects with a reproducible shuffle, so different scenes
of one object never leak between groups. Counts may differ by a few images
because objects are kept intact. Unmapped objects, missing metadata, and
objects matching multiple groups are excluded and counted in the console
summary. If any group has no eligible samples, the script stops before
writing CSVs. Adjust the mapping for your experimental definition of each
YCB group and inspect the reported counts before training.

This script does not create train/validation/test splits within a group.
