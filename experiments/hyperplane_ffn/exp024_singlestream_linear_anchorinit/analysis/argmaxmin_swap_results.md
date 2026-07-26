# argmax/argmin anchor-projection frozen swap: HyperplaneMHL -> FastMHL

Per table row i: anchor_a = argmax(hyperplane_weight[t,i,:]), anchor_b = argmin.
Transfer tables (weights) unchanged, drop bias. FastMHL addressing d_i = x[a]-x[b].
No retraining. Frozen eval, val bpb through the untouched training harness (eval_steps=20).
All 4 loads verified 0 missing / 0 unexpected.

Bit-flip = fraction of address bits where sign(<w_i,x>+b_i) != sign(x[a]-x[b])
on a batch of real val activations, averaged per site.

| run | baseline bpb | anchor-swap bpb | Delta | flip qk | flip v | flip out |
|---|---|---|---|---|---|---|
| exp023 (random-init, dense hyperplanes) | 1.2188 | 4.2378 | +3.0190 | 0.4232 | 0.4260 | 0.4105 |
| exp024 (anchor-init, ~2-sparse hyperplanes) | 1.2150 | 2.6691 | +1.4541 | 0.1566 | 0.1448 | 0.1477 |

Interpretation: anchor-init hyperplanes stay ~2-sparse, so a 2-coordinate anchor
reproduces the learned address well (~15% bit flip) and the frozen swap only
costs +1.45 bpb. Random-init hyperplanes are dense; the best 2 coords capture almost
nothing (~42% flip, near the 50% chance floor) and the swap costs +3.02 bpb.
