"""How much of the SCORED validation text is verbatim present in training?

val_holdout_check.py answered "is there any overlap at all" over a deliberately generous
3,000-document superset. This answers the question that actually bears on the numbers: of the
val text the published bpb was computed on, what FRACTION of characters is also in train?

Document counts are a poor metric here -- a single shared boilerplate footer flags a whole
document. So this measures CHARACTER COVERAGE: mark every position of a scored val document
that falls inside an 80-char window also present in training, and report the covered
fraction. Boilerplate shows up as a few covered percent; genuine leakage of a memorised
document shows up as that document being ~100% covered.

    python val_overlap_scored.py
"""
import glob
import hashlib
import json
import os

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser('~/.cache/nanochat/base_data_climbmix')
SHINGLE, STRIDE = 80, 16          # finer stride than the coarse scan, for coverage accuracy
# Scored slices: bs12 x 10 = 61,440 tokens, bs48 x 10 = 245,760 tokens. At the measured
# 4.661 bytes/token that is ~286k and ~1.145M characters. The bestfit packer draws from a
# 1000-doc buffer, so take the first 1200 docs as a safe superset and report both cut-offs.
N_DOCS = 1200
CHARS_BS12 = 61440 * 4.661
CHARS_BS48 = 245760 * 4.661


def main():
    paths = sorted(glob.glob(os.path.join(DATA, '*.parquet')))
    train_paths, val_path = paths[:-1], paths[-1]

    vf = pq.ParquetFile(val_path)
    docs, cum = [], 0
    for rg in range(vf.num_row_groups):
        for t in vf.read_row_group(rg).column('text').to_pylist():
            docs.append(t)
            cum += len(t)
            if len(docs) >= N_DOCS:
                break
        if len(docs) >= N_DOCS:
            break
    # cumulative char offset of each doc, to locate the two scored cut-offs
    offs, run = [], 0
    for t in docs:
        offs.append(run)
        run += len(t)
    n_bs12 = sum(1 for o in offs if o < CHARS_BS12)
    n_bs48 = sum(1 for o in offs if o < CHARS_BS48)
    print(f'val prefix {len(docs)} docs / {run:,} chars; scored cut-offs: '
          f'bs12 ~ first {n_bs12} docs, bs48 ~ first {n_bs48} docs')

    # val shingle -> list of (doc_idx, pos)
    index = {}
    for i, t in enumerate(docs):
        for p in range(0, max(0, len(t) - SHINGLE), STRIDE):
            index.setdefault(t[p:p + SHINGLE], []).append((i, p))
    print(f'indexed {len(index):,} distinct {SHINGLE}-char shingles (stride {STRIDE})')

    covered = [bytearray(len(t)) for t in docs]
    exact = {hashlib.sha1(t.encode('utf-8', 'ignore')).hexdigest(): i
             for i, t in enumerate(docs)}
    exact_hits = []
    for p in train_paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg).column('text').to_pylist():
                h = hashlib.sha1(t.encode('utf-8', 'ignore')).hexdigest()
                if h in exact:
                    exact_hits.append(dict(val_doc=exact[h], shard=os.path.basename(p),
                                           chars=len(t), preview=t[:150]))
                for q in range(0, max(0, len(t) - SHINGLE), STRIDE):
                    hit = index.get(t[q:q + SHINGLE])
                    if hit:
                        for (i, pos) in hit:
                            covered[i][pos:pos + SHINGLE] = b'\x01' * SHINGLE
        print(f'  scanned {os.path.basename(p)}', flush=True)

    per_doc = [sum(c) / max(1, len(c)) for c in covered]

    def stats(n):
        tot = sum(len(docs[i]) for i in range(n))
        cov = sum(sum(covered[i]) for i in range(n))
        heavy = [i for i in range(n) if per_doc[i] > 0.5]
        return dict(docs=n, chars=tot, covered_chars=cov, covered_frac=cov / max(1, tot),
                    docs_over_50pct_covered=len(heavy), heavy_docs=heavy[:10],
                    docs_with_any_overlap=sum(1 for i in range(n) if per_doc[i] > 0))

    s12, s48, sall = stats(n_bs12), stats(n_bs48), stats(len(docs))
    for name, s in (('bs12 scored slice', s12), ('bs48 scored slice', s48),
                    ('full 1200-doc prefix', sall)):
        print(f'\n{name}: {s["docs"]} docs, {s["chars"]:,} chars')
        print(f'  characters also present verbatim in train: {s["covered_chars"]:,} '
              f'({100*s["covered_frac"]:.3f}%)')
        print(f'  docs with ANY overlap: {s["docs_with_any_overlap"]}  |  '
              f'docs >50% covered (real duplicates): {s["docs_over_50pct_covered"]}')
    print(f'\nexact whole-document duplicates in the 1200-doc prefix: {len(exact_hits)}')
    for h in exact_hits:
        loc = ('INSIDE bs12 slice' if h['val_doc'] < n_bs12 else
               'inside bs48 slice' if h['val_doc'] < n_bs48 else 'outside both scored slices')
        print(f'   val doc {h["val_doc"]} ({h["chars"]:,} chars) <- {h["shard"]}  [{loc}]')
        print(f'     {h["preview"][:110]!r}')

    out = dict(shingle_len=SHINGLE, stride=STRIDE, val_file=os.path.basename(val_path),
               train_files=[os.path.basename(p) for p in train_paths],
               n_docs_indexed=len(docs), bs12_docs=n_bs12, bs48_docs=n_bs48,
               bs12=s12, bs48=s48, full_prefix=sall,
               exact_duplicates=exact_hits,
               top_covered_docs=sorted(range(len(docs)), key=lambda i: -per_doc[i])[:10],
               top_covered_fracs=sorted(per_doc, reverse=True)[:10])
    with open(os.path.join(HERE, 'val_overlap_scored.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {HERE}/val_overlap_scored.json')


if __name__ == '__main__':
    main()
