"""Generate out_proj sweep sub-experiments forked from exp314.

Grid: input_nap in [6, 8, 10, 12] x tph in [128, 256, 512, 1024, 2048]
      x output_nap in [32, 128, 256]
Filter: keep configs where 6-layer total bit params <= 700M.
       bits_per_layer = tph * 2^input_nap * output_nap.

Each sub-dir gets:
  - config.json  (exp314 baseline with out_input_nap/out_tph/out_output_nap
                  overridden, n_steps=10000)
  - train.py     (copy of exp314/train.py with the two ../.. path-climbs
                  updated to ../../.. since sub-dirs are one level deeper)

Also emits configs.txt listing sub-dir names in size-ascending (total-bits)
order for the driver script.
"""
import json, os, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
EXP314 = os.path.normpath(os.path.join(HERE, '..', 'exp314_dom_canon_sdpa'))
EXP314_TRAIN = os.path.join(EXP314, 'train.py')

with open(os.path.join(EXP314, 'config.json')) as f:
    base_cfg = json.load(f)

INAP_GRID  = [6, 8, 10, 12]
TPH_GRID   = [128, 256, 512, 1024, 2048]
ONAP_GRID  = [32, 128, 256]

N_LAYERS = base_cfg['num_layers']
BUDGET   = 700_000_000

rows = []
for inap, tph, onap in itertools.product(INAP_GRID, TPH_GRID, ONAP_GRID):
    bits_per_layer = tph * (1 << inap) * onap
    total_bits = bits_per_layer * N_LAYERS
    if total_bits > BUDGET:
        continue
    rows.append((total_bits, inap, tph, onap, bits_per_layer))

rows.sort()  # ascending by total_bits

with open(EXP314_TRAIN) as f:
    train_src_orig = f.read()

# exp314 lives at transformer_exps/<exp>/. Sweep sub-dirs live at
# transformer_exps/sweep_out_proj/<exp>/ (one level deeper), so the two
# '../..' path climbs need an extra '..'.
train_src = train_src_orig.replace(
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))",
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))",
).replace(
    "os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt')",
    "os.path.join(os.path.dirname(__file__), '..', '..', '..', 'workbooks', 'fineweb_texts.txt')",
)
assert train_src != train_src_orig, 'path-climb patch did not apply'

dir_names = []
for total_bits, inap, tph, onap, bpl in rows:
    dir_name = f'cfg_in{inap:02d}_tph{tph:04d}_o{onap:03d}'
    sub = os.path.join(HERE, dir_name)
    os.makedirs(sub, exist_ok=True)

    cfg = dict(base_cfg)
    cfg['exp_name'] = dir_name
    cfg['description'] = (
        f'out_proj sweep: input_nap={inap}, tph={tph}, output_nap={onap}. '
        f'out_proj bits: {bpl/1e6:.2f}M/layer, {total_bits/1e6:.1f}M total. '
        f'Forked from exp314, 10K steps, bs=8.'
    )
    cfg['out_input_nap']  = inap
    cfg['out_tph']        = tph
    cfg['out_output_nap'] = onap
    cfg['n_steps']        = 10000
    with open(os.path.join(sub, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(sub, 'train.py'), 'w') as f:
        f.write(train_src)
    dir_names.append(dir_name)

with open(os.path.join(HERE, 'configs.txt'), 'w') as f:
    f.write('\n'.join(dir_names) + '\n')

print(f'Generated {len(rows)} configs under {HERE}')
print(f'{"#":>3} {"in":>3} {"tph":>5} {"out":>4} {"M/layer":>8} {"M total":>8}  dir')
for i, (total, inap, tph, onap, bpl) in enumerate(rows, 1):
    print(f'{i:>3} {inap:>3} {tph:>5} {onap:>4} {bpl/1e6:>8.2f} {total/1e6:>8.1f}  '
          f'cfg_in{inap:02d}_tph{tph:04d}_o{onap:03d}')
