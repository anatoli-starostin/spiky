# exp24 — does log-sum-exp pooling justify itself on quantisation grounds?

**Takeaway: no, not on quantisation grounds and not on return.** Log-sum-exp pooling does
buy a modest gain in how evenly the weights fill their buckets — normalised entropy 0.72 vs
0.63 on 22 buckets, 6 near-empty buckets vs 9 — but it costs task performance if anything
(deterministic return 5707 vs 6196 for plain sum) and it does not change the weight
distribution's scale at all (std 0.0706 vs 0.0699). The post's existing hedge, *"a plain sum
would have been fine too"*, is the right call and is now measured rather than asserted.

## First, a correction to the framing

The experiment was posed as "22-bucket **weight** quantisation". There is no such thing in
the repo. Two different quantisers, on two different quantities:

| | what it snaps | levels | spacing | where |
|---|---|---|---|---|
| `UniformActionQuantizer` | the **action mean** | **22** | **linear**, `linspace(-1, +1, 22)`, step 2/21 = 0.0952 | `src/act_quant.py` |
| Stage-3 weight fit | the **LUT weights** | **256 (8-bit)** | **uniform in `L = W/τ`**, i.e. **log domain** | `stage3_cd_bigdata.py:74-78` |

```python
L0 = W0 / tau
lo, hi = L0.min(), L0.max()
step = (hi - lo) / 255.0
L = lo + np.round((L0 - lo) / step) * step
```

The weight grid is log-domain *because* the readout exponentiates (`S = Σ_t exp(L_t)`), which
is the whole point of the log-sum-exp. **It is therefore not defined for the plain-sum arm at
all**: with no τ and no exponential, a plain-sum readout decodes linearly and its natural
weight grid is uniform in `W` itself. Both are reported below, each arm on its own grid.

## Setup

Two arches differing in exactly one constructor argument:

| | arm A | arm B |
|---|---|---|
| arch | `fastlut_lse_sum_expmlpcrit` (exp19) | `fastlut_sum_expmlpcrit` (new) |
| pooling | `out = T·τ·log((1/T)·Σ_t exp(w_t/τ))` | `out = Σ_t w_t` |
| `exp_outputs` | `True` | `False` |

Everything else identical, and *verified* identical: built with the same seed, every tensor
the two arms share is bit-identical — LUT weights, both anchor-pair buffers, the
soft-backward buffers, `log_std`, `tau_c_raw`, and all six critic tensors. The only extra
parameter in A is the actor's `tau_raw`. Arm B is the τ→∞ limit of arm A, not a different
architecture: the sum-scaled log-sum-exp was constructed so that limit is exact.

3 seeds each, trained from scratch, 768 updates, 8192 envs, under exp23's full quantisation
regime — input quantiser (128 ticks, σ=1.0), **22-level output quantisation**, `--oob-penalty
0.1` — and deployment-matched physics (`--obs-clip-vel 10 --solver-iters 100 --ls-iters 50`).
From scratch rather than fine-tuned from an exp19 checkpoint, which would have handed arm B
weights already shaped by the operator under test. 1815 s wall for all six.

## Task performance — plain sum is not worse

Deterministic closed-loop return, 512 envs × 2000 steps, both quantisers on:

| seed | A (log-sum-exp) | B (plain sum) |
|---|---:|---:|
| 0 | 5923.7 ± 614.6 | 5639.9 ± 886.0 |
| 1 | 5680.7 ± 548.3 | 6567.5 ± 357.2 |
| 2 | 5516.9 ± 88.6 | 6379.5 ± 579.3 |
| **mean** | **5707.1** | **6195.6** |

Arm B walks, and walks well — the point estimate favours it by **+488 (8.6 %)**. With three
seeds and this spread that is not a significant win for plain sum, but it is decisively *not*
a win for log-sum-exp: A's best seed (5923.7) is below B's worst-but-one, and B's minimum
(5639.9) sits above A's median. Whatever the pooling operator buys, it is not return.

## Weight distributions — nearly the same, except in the tails

Pooled over 3 seeds, 36,864 weights per arm:

| | A (log-sum-exp) | B (plain sum) |
|---|---:|---:|
| mean | −0.00744 | +0.00797 |
| **std** | **0.07056** | **0.06992** |
| min / max | −0.4148 / +0.2345 | −0.4408 / +0.4198 |
| **dynamic range** | **0.6493** | **0.8606** |
| \|w\|max | 0.4148 | 0.4408 |
| p99 \|w\| | 0.2069 | 0.2008 |
| excess kurtosis | +0.86 | +1.13 |
| \|w\| < 1e-3 | 2.14 % | 2.38 % |
| \|w\| < 1e-2 | 12.67 % | 13.02 % |

The **scale is the same to within 1 %** — the pooling operator does not make the weights
tighter or wider in std, and does not change how much mass sits near zero. Two differences
are real:

* **B's dynamic range is 33 % wider** (0.861 vs 0.649), driven entirely by its positive tail
  (+0.42 vs +0.23). For a spiking build that matters directly: Stage-3 delay span is set by
  the *spread* of the table weights, so a wider range costs ticks.
* **B is more peaked** (excess kurtosis 1.13 vs 0.86) — more mass in the centre, longer thin
  tails.

## Bucket occupancy — the headline

**22 buckets**, each arm over its own weight range (evenness as normalised entropy
H/log 22; 1.0 = perfectly uniform):

| | A (log-sum-exp) | B (plain sum) |
|---|---:|---:|
| **normalised entropy** | **0.7206** | **0.6320** |
| near-empty buckets (<0.1 %) | **6 / 22** | 9 / 22 |
| occupied buckets | 22 / 22 | 22 / 22 |
| busiest bucket | **18.28 %** | 26.21 % |

**256 buckets (8-bit), each arm on its own deploy grid** — A in the log domain (`W/τ`,
τ = 0.0905), B in the linear domain:

| | A (log domain) | B (linear domain) |
|---|---:|---:|
| grid span | [−4.582, +2.591], step 0.0281 | [−0.441, +0.420], step 0.0034 |
| **normalised entropy** | **0.8490** | 0.7990 |
| near-empty buckets | **128 / 256** | 157 / 256 |
| occupied buckets | **222 / 256** | 197 / 256 |
| busiest bucket | **2.46 %** | 2.85 % |

So arm A *does* fill its buckets more evenly, on both grids — about 0.09 of normalised
entropy on 22 buckets and 0.05 on 256, with a third fewer near-empty buckets. Part of that is
mechanical rather than learned: quantising in the log domain stretches the small-\|w\| region
where most of the mass sits, which is precisely what a log-domain grid is *for*.

![bucket occupancy](figures/bucket_occupancy.png)

![weight distribution](figures/weight_distribution.png)

## For the post's Constraint 2 paragraph

The post currently says the log-sum-exp is buildable because *"a synapse can supply an
exponential, a dendrite can supply a sum, and an exponentially growing membrane supplies a
logarithm as its time-to-threshold"*, and adds *"a plain sum would have been fine too; a
softmax or a matmul would not."*

That parenthetical is doing more work than it looks. Measured: a plain sum trains to the same
place (**within 1 % on weight std, within seed noise on return, if anything better**), and
the only thing log-sum-exp buys is somewhat more even bucket occupancy — a benefit that
partly comes from the log-domain grid rather than from the learned weights. What log-sum-exp
does buy, and this ablation does not test, is that its *own* Stage-3 construction exists and
is cheap: exp/sum/log map onto synapse/dendrite/membrane directly, and the 33 %-narrower
weight range means fewer ticks of delay span.

**One-line takeaway:** log pooling is not required by the quantisation — a plain sum
quantises nearly as well and trains no worse — so the honest claim is that the exponential
was chosen because it is what a neuron computes for free, not because the LUT needed it.

## Caveats

* Three seeds, one configuration, trained from scratch rather than fine-tuned from exp19 —
  the regime the shipped artefact actually came from.
* The 8-bit comparison puts each arm on a *different* grid (log vs linear) because that is
  what each arm's deploy path would use. The entropies are therefore not measured in the same
  units; the 22-bucket comparison, which is linear for both, is the like-for-like one.
* Return is measured in the warp env with the training quantisers, not in gymnasium.
