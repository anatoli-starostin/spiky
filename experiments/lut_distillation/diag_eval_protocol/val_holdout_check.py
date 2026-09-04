"""Is the validation data genuinely held out from training? Read the data, don't assume.

Three checks, from cheapest to strongest:

  1. SHARD LEVEL. `_document_batches` does
         parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
     i.e. a POSITIONAL split: train is every parquet file but the last, val is the last one
     alone. Necessary but not sufficient -- identical documents could still sit in both files.

  2. EXACT DUPLICATES. SHA1 every training document and check whether any validation document
     in the SCORED prefix appears verbatim in training.

  3. NEAR DUPLICATES. Exact matching misses reformatted or partially-overlapping copies, so
     also shingle the scored val prefix (80-char windows, stride 32) and stream every training
     document looking for a shared shingle. An 80-character verbatim overlap is far beyond
     coincidence in natural text, so any hit is real contamination.

Scope note: checks 2 and 3 cover the val documents that feed the SCORED slices (the first
61,440 tokens for bs12 runs, 245,760 for bs48). That is deliberate -- those are the only val
tokens any published number was ever computed on. The rest of the val shard is not scored by
anything, so contamination there could not have affected a result.

    python val_holdout_check.py
"""
import glob
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser('~/.cache/nanochat/base_data_climbmix')
SHINGLE, STRIDE = 80, 32
# Superset of the documents feeding the scored slices. The bestfit packer draws from a
# 1000-document buffer, so the first 245,760 tokens (~1.15M chars) come from within roughly
# the first ~1.5k documents; 3000 is a deliberately generous over-estimate.
N_VAL_DOCS = 3000


def sha1(s):
    return hashlib.sha1(s.encode('utf-8', 'ignore')).hexdigest()


def shingles(text):
    for i in range(0, max(0, len(text) - SHINGLE), STRIDE):
        yield text[i:i + SHINGLE]


def main():
    paths = sorted(glob.glob(os.path.join(DATA, '*.parquet')))
    train_paths, val_paths = paths[:-1], paths[-1:]
    print('SHARD SPLIT (positional: train = all but last file, val = last file)')
    for p in train_paths:
        print(f'  train  {os.path.basename(p)}  rows {pq.ParquetFile(p).metadata.num_rows:,}')
    for p in val_paths:
        print(f'  VAL    {os.path.basename(p)}  rows {pq.ParquetFile(p).metadata.num_rows:,}')
    assert not (set(train_paths) & set(val_paths)), 'train/val file overlap!'
    print(f'  file-level overlap: NONE ({len(train_paths)} train files, '
          f'{len(val_paths)} val file)\n')

    # --- collect the scored val prefix, in the order the loader reads it -----------------
    vf = pq.ParquetFile(val_paths[0])
    val_docs, chars = [], 0
    for rg in range(vf.num_row_groups):
        for t in vf.read_row_group(rg).column('text').to_pylist():
            val_docs.append(t)
            chars += len(t)
            if len(val_docs) >= N_VAL_DOCS:
                break
        if len(val_docs) >= N_VAL_DOCS:
            break
    val_hashes = {sha1(t) for t in val_docs}
    val_shingles = {}
    for i, t in enumerate(val_docs):
        for s in shingles(t):
            val_shingles.setdefault(s, i)
    print(f'val prefix: {len(val_docs):,} documents, {chars:,} chars '
          f'(~{chars/4.7:,.0f} tokens at 4.7 B/token -- the scored slices are 61,440 and '
          f'245,760 tokens, so this is a superset)')
    print(f'  {len(val_hashes):,} distinct doc hashes, {len(val_shingles):,} distinct '
          f'{SHINGLE}-char shingles\n')

    # --- stream ALL training documents ----------------------------------------------------
    exact_hits, shingle_hits, n_train_docs, train_chars = [], [], 0, 0
    for p in train_paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg).column('text').to_pylist():
                n_train_docs += 1
                train_chars += len(t)
                if sha1(t) in val_hashes:
                    exact_hits.append(dict(shard=os.path.basename(p), rg=rg,
                                           preview=t[:120]))
                for s in shingles(t):
                    if s in val_shingles:
                        shingle_hits.append(dict(shard=os.path.basename(p), rg=rg,
                                                 val_doc=val_shingles[s], shingle=s))
                        break        # one hit per train doc is enough to flag it
        print(f'  scanned {os.path.basename(p)}: {n_train_docs:,} docs so far, '
              f'{len(exact_hits)} exact, {len(shingle_hits)} near', flush=True)

    print(f'\ntrain corpus scanned: {n_train_docs:,} documents, {train_chars:,} chars')
    print(f'EXACT duplicate documents (val prefix found verbatim in train): {len(exact_hits)}')
    print(f'NEAR duplicates ({SHINGLE}-char verbatim overlap): {len(shingle_hits)}')
    for h in shingle_hits[:5]:
        print(f'    {h["shard"]} rg{h["rg"]} ~ val doc {h["val_doc"]}: {h["shingle"][:70]!r}')

    out = dict(
        data_dir=DATA,
        train_files=[os.path.basename(p) for p in train_paths],
        val_file=os.path.basename(val_paths[0]),
        split_rule='positional: parquet_paths[:-1] train, parquet_paths[-1:] val',
        file_level_overlap=False,
        val_prefix_docs=len(val_docs), val_prefix_chars=chars,
        train_docs_scanned=n_train_docs, train_chars_scanned=train_chars,
        shingle_len=SHINGLE, shingle_stride=STRIDE,
        exact_duplicate_count=len(exact_hits),
        near_duplicate_count=len(shingle_hits),
        exact_examples=exact_hits[:5], near_examples=shingle_hits[:5],
        verdict=('CLEAN' if not exact_hits and not shingle_hits else 'CONTAMINATION FOUND'))
    with open(os.path.join(HERE, 'val_holdout.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nVERDICT: {out["verdict"]}')
    print(f'wrote {HERE}/val_holdout.json')


if __name__ == '__main__':
    main()
