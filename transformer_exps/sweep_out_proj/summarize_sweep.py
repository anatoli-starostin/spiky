"""Summarize out_proj sweep: rank sub-experiments by best_val_loss + bit budget."""
import json, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))

rows = []
for sub in sorted(glob.glob(os.path.join(HERE, 'cfg_*'))):
    cfg_path = os.path.join(sub, 'config.json')
    sum_path = os.path.join(sub, 'summary.json')
    if not os.path.exists(cfg_path):
        continue
    cfg = json.load(open(cfg_path))
    cid = cfg['exp_name'].split('_')[1]
    inap = cfg['out_input_nap']
    tph = cfg['out_tph']
    onap = cfg['out_output_nap']
    bpl = tph * (1 << inap) * onap
    total = bpl * cfg['num_layers']

    if os.path.exists(sum_path):
        s = json.load(open(sum_path))
        val = s.get('best_val_loss')
        hrs = s.get('training_time_hours')
        status = 'done'
    else:
        val = None
        hrs = None
        status = 'pending'
    rows.append((val if val is not None else float('inf'),
                 cid, inap, tph, onap, bpl, total, hrs, status))

rows.sort()
print(f'{"rank":<4} {"id":<3} {"in":>3} {"tph":>5} {"out":>4} {"M total":>8} '
      f'{"val_loss":>9} {"hrs":>5}  status')
for i, (val, cid, inap, tph, onap, bpl, tot, hrs, st) in enumerate(rows, 1):
    v = f'{val:.4f}' if val != float('inf') else '  n/a  '
    h = f'{hrs:.2f}' if hrs is not None else '  -  '
    print(f'{i:<4} {cid:<3} {inap:>3} {tph:>5} {onap:>4} {tot/1e6:>7.1f}M '
          f'{v:>9} {h:>5}  {st}')
