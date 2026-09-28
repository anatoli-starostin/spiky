"""Force a verbose rebuild so -Xptxas -v is captured, and print the raw refusal messages."""
import os
import shutil
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ts  # noqa: E402

if os.path.isdir(ts.BUILD):
    shutil.rmtree(ts.BUILD)          # force a real compile so ptxas -v is emitted
m = ts.mod(verbose=True)
print('BUILD DONE')

tab = torch.zeros(256 * 256, 1024, device='cuda', dtype=torch.int8)
cells = torch.zeros(512, 1, 256, 3, device='cuda', dtype=torch.uint8)
print('\nRAW REFUSAL MESSAGES')
for G, L, Bt in ts.CONFIGS:
    if ts.budget(G, L, Bt)['fits_smem']:
        continue
    try:
        ts.run(tab, cells, 1024, G, L, Bt)
        print(f'  G={G} L={L} Bt={Bt}: NO ERROR (unexpected)')
    except Exception as ex:
        first = str(ex).splitlines()[0]
        print(f'  G={G} L={L} Bt={Bt}: {first[:190]}')
