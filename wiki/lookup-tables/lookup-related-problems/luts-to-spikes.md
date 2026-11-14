# LUTs to spikes conversion (lutwalker2d)

The lutwalker2d experiment: a lookup-table policy trained with reinforcement learning to control the Walker2d robot, and then a classic spiking network built from the trained table. Everything on this page is told in the write-up [*A handcrafted spiking network that matches a quantised Walker2d policy*][post] (the post); numbers are the post's, and no run was repeated for this page, so every result below is *claimed*. The post was also published as a web page, which has since been taken down.

## Idea / model

**Problem.** Can a policy that is a lookup table be turned into a spiking network that behaves like it, without training the network by gradient descent? The target was the Gymnasium `Walker2d` task (17 observation numbers, 6 action numbers).

**The teacher.** The policy is an anchor-pair table model ([`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/lutorch/fast_multi_head_lut.py#L754)): 32 tables with 6 anchor pairs each, so a table address is made of 6 comparisons $x[a_i]>x[b_i]$ between observation coordinates and the policy never does arithmetic on the observation. The addressed rows of the 32 tables are combined by the log-sum-exp readout of [readout 3.1](../lookup-table-primitives/readout.md#31-log-sum-exp-over-tables):

$$\mu_o=N\tau\log\Big(\frac{1}{N}\sum_{t=1}^{N}e^{W_t[c_t][o]/\tau}\Big)$$

Reading: the number of tables is $N=32$ and the entry of output $o$ in the row $c_t$ addressed in table $t$ is $W_t[c_t][o]$ and $\tau$ is a learned temperature. The action is $\mu_o$ clipped at minus one and plus one. The policy was trained with a GPU-resident PPO written for this work (torch with MuJoCo-Warp), using the stabilisation recipe of [exp05](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/exp05_ppo-truncbootstrap-retnorm-kl) (return normalisation and exact truncation bootstrap, found on a plain MLP policy), see [exp19](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32).

**Quantisation at the interfaces.** A spiking network has discrete inputs and outputs, so the policy was fine-tuned with quantisers in the loop ([exp23](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/exp23_qat_obs_quant)): the observation goes into 128 buckets through one monotone map shared by all coordinates (shared because the table compares coordinates with each other), and the action is snapped to 22 levels. The table entries themselves stay in float32.

**The network** is built in closed form on the SpNet spiking engine (described in [SpNet](../../spnet/README.md); code: [`src/spiky/spnet`](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet)), in three stages (2,889 neurons and 25,953 synapses in total):

1. *Order detection.* Each coordinate becomes a spike whose time encodes its value (larger value, earlier spike); winner-take-all comparator pairs decide the 192 address bits from the order of arrival.
2. *Lookup.* One neuron per table cell (2,048 cells); a cell fires only when all 6 address spikes reach it.
3. *Readout.* The 32 selected cells send spikes whose synaptic weights are $e^{W_t/\tau}$ into an output neuron whose membrane grows exponentially; the time it takes to reach the threshold is a logarithm of the summed input, so the log-sum-exp is computed by the neuron itself:

$$t_\text{cross}=\tau_\text{eff}\Big(\log\theta-\log\sum_{t}e^{W_t/\tau}\Big)$$

Reading: the firing threshold is $\theta$ and the time constant of the exponential growth is $\tau_\text{eff}$ and so a larger table value gives an earlier spike, and the action is decoded from the spike time. The network needs no external clock: a completion neuron fires when the last input has arrived.

The only fitted part is the 12,288 readout weights, put on an 8-bit logarithmic grid by coordinate descent against the teacher's quantised actions (153,600 teacher input-output pairs).

## Why it matters / what was found

*Claimed* in the post, closed-loop Walker2d, about 520 episodes per build, deterministic actions, 256 environments times 2 seeds:

| build | mean return | standard error |
| --- | ---: | ---: |
| software teacher (quantised policy) | 6253.4 | 36.3 |
| spiking network, weights as in the table | 6119.4 | 40.9 |
| spiking network, weights fitted | 6312.2 | 26.6 |

- The fitted network is *statistically indistinguishable* from the teacher: the difference is plus 58.8 with an error of 45.0, which is 1.31 standard errors. The post says explicitly that this is not evidence that the network is better than the teacher.
- The fit mattered: the network built straight from the table scored 164.8 below the teacher, because of a one-sided rounding bias in the decoded actions (about one fifth of the actions were one level too low). The fit brought exact agreement with the teacher's 22-level actions to 98.58% on held-out states.
- About 155 integer ticks per decision; the readout stage was made short by moving the stored value from a spike delay to a synaptic weight (see the first dead end).

## Dead ends

**Dead end:** value stored as a spike delay. The first build encoded each table value as a delay; it needed episodes of about 309 ticks against about 155 for the weight encoding, and was replaced. It is still served by the demo stand as a separate actor.

**Dead end:** dedicated tie detectors. Quantisation creates many ties between coordinates; 136 extra neurons handled them until one extra tick of delay on one comparator rail reproduced the same rule at no cost.

**Dead end:** gradient-based fitting of the readout weights. The post reports two failed attempts (a straight-through optimiser diverged, a closed-form model was unfaithful to the network until its timing phase was pinned) before the gradient-free coordinate descent.

**Dead end:** recalibrating the decode offset alone. An on-policy offset shift flipped one output dimension wholesale, and the post reports minus 175 for it in the walker evaluation. Related trap: the fitted weights and the decode offset belong together; the fitted weights with the old offset gave minus 197.0 in the same comparison where the matching pair gives plus 58.8, and a wrong conclusion was drawn from it before that was noticed.

**Dead end:** exponential critic readout and the soft maximum. In [exp19](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32/README.md) the critic got the same log-sum-exp readout; it was harmless but did nothing (3 seeds each), and in the runs exp17 to exp19 the learned temperature moved up, toward the plain sum, never toward the maximum. See also [readout 3.1](../lookup-table-primitives/readout.md#31-log-sum-exp-over-tables).

**Evaluation traps** listed by the post: episodes shorter than the 1000-step limit give a biased (censored) average, and 65 episodes are too few (standard error about 110) for effects of 165.

## Open ideas

The post names none.

## Sources

- The post: [`WALKER2D_SPIKING_WRITEUP.md`][post]. Its web version was published on GitHub Pages and is currently offline; the last full published version is the [`walker2d-spiking` page at commit `e829dbc6` of the `gh-pages` branch][web].
- Branch map and runbook: [`experiments/walker2d-lut/README.md`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/README.md) and [`REPRODUCE.md`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/REPRODUCE.md).
- Construction scripts: [`walker2d-spiking/`](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/walker2d-spiking); the live demo stand: [`landing/walker2d-viz`](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/landing/walker2d-viz).
- The earlier research tree (other architectures, SAC, failed fits) is on the branch [`research/walker2d-lut`](https://github.com/anatoli-starostin/spiky/tree/04055f0c264cfd43b8921fc2892e8b4098c218c5); it was not reviewed for this page.

[post]: https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/walker2d-spiking/WALKER2D_SPIKING_WRITEUP.md
[web]: https://github.com/anatoli-starostin/spiky/blob/e829dbc6aeeba5a0189f290fad25d4b10d2fd469/walker2d-spiking/index.html
