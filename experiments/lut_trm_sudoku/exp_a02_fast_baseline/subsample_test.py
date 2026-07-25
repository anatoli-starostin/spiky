#!/usr/bin/env python3
"""Create a ~5k-puzzle subsampled TEST set for the fast-iteration TRM run (exp_a02).

The full Sudoku-Extreme test set is 422,786 puzzles → each eval is ~826 batches (minutes).
For fast A/B iteration on LUT swaps we only need a stable signal, so we slice the first
N=5000 test puzzles into a new dataset dir. The full dataset is left untouched; train/ is
symlinked (unchanged full augmented train set), only test/ is subsampled.

Test format (verified): 1 example per puzzle, 1 puzzle per group, so:
  inputs[:N], labels[:N], puzzle_identifiers[:N], and puzzle_indices/group_indices = arange(N+1).
"""
import os, json, shutil
import numpy as np

N = 5000
SRC = os.path.expanduser("~/projects/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000")
DST = os.path.expanduser("~/projects/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000-testsub5k")

os.makedirs(DST, exist_ok=True)

# train/ : symlink the untouched full augmented train split
train_link = os.path.join(DST, "train")
if not os.path.islink(train_link) and not os.path.exists(train_link):
    os.symlink(os.path.join(SRC, "train"), train_link)

# test/ : sliced copy
dst_test = os.path.join(DST, "test")
os.makedirs(dst_test, exist_ok=True)
src_test = os.path.join(SRC, "test")

inputs = np.load(os.path.join(src_test, "all__inputs.npy"))
labels = np.load(os.path.join(src_test, "all__labels.npy"))
pids   = np.load(os.path.join(src_test, "all__puzzle_identifiers.npy"))
n = min(N, inputs.shape[0])

np.save(os.path.join(dst_test, "all__inputs.npy"), inputs[:n])
np.save(os.path.join(dst_test, "all__labels.npy"), labels[:n])
np.save(os.path.join(dst_test, "all__puzzle_identifiers.npy"), pids[:n])
np.save(os.path.join(dst_test, "all__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))
np.save(os.path.join(dst_test, "all__group_indices.npy"), np.arange(n + 1, dtype=np.int32))

# metadata: same as source test but with the reduced counts
meta = json.load(open(os.path.join(src_test, "dataset.json")))
meta["total_groups"] = n
meta["total_puzzles"] = n
json.dump(meta, open(os.path.join(dst_test, "dataset.json"), "w"))

print(f"Wrote {n}-puzzle test subset to {dst_test}")
print(f"train/ symlinked -> {os.path.realpath(train_link)}")
print("metadata:", meta)
