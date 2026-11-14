# SpNet: the math

The equations that SpNet computes, read from its CUDA kernels and set next to the original program of Eugene Izhikevich, so that every statement can be checked at both ends. Nothing was run for this page: "agree" means that the code lines perform the same operation at the stated parameters. The [architecture document](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/doc/spiky/spiky_highlevel_architecture.md) says that SpNet implements the model of the polychronization paper; the comparison below is what shows how far the code matches the original program, especially for the STDP rule.

## The original

- Page: [Polychronization: Computation With Spikes](https://www.izhikevich.org/publications/spnet.htm) (Neural Computation 18:245-282, 2006; the architecture document cites it as 2003). Archived copy: [web.archive.org](https://web.archive.org/web/20230329005309/http://izhikevich.org/publications/spnet.htm).
- Code: [`spnet.m`](https://www.izhikevich.org/publications/spnet.m) (MATLAB) and [`spnet.cpp`](https://www.izhikevich.org/publications/spnet.cpp) (C), linked from that page; the paper is [`spnet.pdf`](https://www.izhikevich.org/publications/spnet.pdf). The page says the two programs "are similar but not equivalent", so the table cites both where they agree.
- The line numbers below refer to these archived copies, which are byte-identical to the files on the site when fetched on 2026-11-14: [`spnet.m`](https://web.archive.org/web/20230607050147/http://izhikevich.org/publications/spnet.m) (header: modified April 16, 2008) and [`spnet.cpp`](https://web.archive.org/web/20230607050057/http://izhikevich.org/publications/spnet.cpp).

## Notation

One tick is 1 ms. A spike of neuron $j$ at tick $t$ arrives at neuron $i$ after $\delta_{ji}$ ticks. The original numbers delays from 1 to 20 and Spiky from 0 (same tick) to 255, so a delay of $d$ ms in the original is $\delta=d-1$ in Spiky; the runtime test uses 0 to 19 for the original's 1 to 20 ([runtime test L75-83](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L75-L83)). Symbols, with the original's values and the Spiky names (defaults in brackets):

| Symbol | Meaning | Original | Spiky |
|---|---|---|---|
| $v$ $u$ | voltage, recovery variable | `v` `u` | per-neuron state |
| $k_2$ $k_1$ $k_0$ | coefficients of the voltage equation | 0.04 5 140 | `cf_2` `cf_1` `cf_0` (same) |
| $a$ | recovery rate | 0.02 excitatory, 0.1 inhibitory | `a` (0.02) |
| $b$ | recovery coupling | 0.2 | `b` (0.2) |
| $c$ | reset voltage | -65 | `c` (-65) |
| $d$ | recovery increment at a spike | 8 excitatory, 2 inhibitory | `d` (8) |
| $\theta$ | spike threshold | 30 | `spike_threshold` (30) |
| $\delta_{ji}$ | delay of the synapse from $j$ to $i$ | 1 to 20 ms | `min_delay` to `max_delay` ticks |
| $w_{ji}$ | synaptic weight | starts at 6 or -5 | `initial_weight` (0) |
| $\rho$ | decay of the STDP traces per tick | 0.95 | `stdp_decay` (0.95) |
| $p$ | potentiation amplitude | 0.1 | `ltp_max` (1.0) |
| $q$ | depression amplitude | 0.12 | `ltd_max` (1.2) |
| $\eta$ | learning rate | 1 | `learning_rate` (0.05) |
| $\gamma$ | constant drift per update | 0.01 | `weight_scaling_cf` (0.01) |
| $\lambda$ | decay of the accumulator per update | 0.9 | `weight_decay` (0.9) |
| $w_{\min}$ $w_{\max}$ | weight bounds | 0 and 10 | `min_weight` `max_weight` (0 and 10) |
| $\Delta_{ji}$ | accumulated STDP change of a synapse | `sd` | internal weight-delta buffer |

## Neuron (per tick)

On each tick, first the reset of a neuron whose voltage reached the threshold on the previous update:

$$v\ge\theta:\quad v\leftarrow c,\quad u\leftarrow u+d$$

then two half-steps for the voltage with the input $I$ held fixed, then the recovery variable with the new voltage:

$$v\leftarrow v+0.5\left(k_2v^2+k_1v+k_0-u+I\right)\quad\text{(twice)}$$

$$u\leftarrow u+a\left(bv-u\right)$$

Reading: the usual Izhikevich simple model with the 0.5 ms integration trick of the original. The input is the sum of the external input and the weights of the spikes that arrive on this tick:

$$I_i(t)=I^{\mathrm{ext}}_i(t)+\sum_{j}w_{ji}s_j(t-\delta_{ji})$$

Here $s_j(t)$ is 1 if neuron $j$ spikes at tick $t$ and 0 otherwise. The current is a pulse: it acts on one update and is cleared afterwards.

## STDP

Let $\tau_j(t)$ be the number of ticks since the most recent spike of neuron $j$ up to and including tick $t$ itself, with 0 on the tick of the spike. Only the most recent spike counts, there is no sum over earlier spikes. The two tables of exponentially decaying values are:

$$P(k)=p\rho^{k},\qquad Q(k)=q\rho^{k}$$

The accumulator $\Delta_{ji}$ of the synapse from $j$ to $i$ changes in two ways. When neuron $i$ spikes at tick $t$ the presynaptic spikes came first, which potentiates:

$$\Delta_{ji}\leftarrow\Delta_{ji}+P\left(\tau_j(t-\delta_{ji}-1)\right)$$

When a spike of $j$ arrives at $i$ on tick $t$ the postsynaptic spike came first, which depresses; a spike of $i$ on this very tick counts:

$$\Delta_{ji}\leftarrow\Delta_{ji}-Q\left(\tau_i(t)\right)$$

Reading: the potentiation of a synapse is the value of the presynaptic trace one tick before its spike would have arrived, the depression is the value of the postsynaptic trace at the arrival. The weights are updated with the symbols of the table:

$$w_{ji}\leftarrow\min\left(w_{\max},\max\left(w_{\min},w_{ji}+\eta\Delta_{ji}+\gamma\right)\right),\qquad\Delta_{ji}\leftarrow\lambda\Delta_{ji}$$

Reading: the weight moves by the accumulated change plus a small constant drift, is clipped, and the accumulator then decays; the accumulator is not reset. The update runs once per 1000 ticks in the original. The runtime test sets the learning rate to 0.1 with the default `ltp_max` 1.0 and `ltd_max` 1.2, which gives the amplitudes of the original:

$$\eta p=0.1,\qquad\eta q=0.12$$

([test](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L75-L83), [spnet.py L28-30](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L28-L30)). The three STDP parameters $\rho$ $p$ $q$ belong to the postsynaptic neuron's type; $\eta$ $\gamma$ $\lambda$ and the bounds belong to the synapse type.

## Original and Spiky, element by element

**agree**: the same operation at matching parameters; **differ**: a different rule or design; **cannot compare**: the two codes do not define the same thing.

| Element | Original | Spiky | Verdict |
|---|---|---|---|
| Voltage and recovery update (two 0.5 steps for the voltage, then the recovery with the new voltage) | [spnet.m L70-72][orig-m]; [spnet.cpp L139-141][orig-c] | [kernels L802-853](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L802-L853); [spnet.h L28-29](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet.h#L28-L29) | agree |
| Spike test and reset: threshold 30, voltage reset to `c`, recovery increased by `d`, at the start of the tick after the update crossed the threshold | [spnet.m L54-56][orig-m]; [spnet.cpp L114-118][orig-c] | [kernels L253](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L253); [kernels L785-788](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L785-L788) | agree. Spiky also has a subtractive reset and a refractory window, both off by default ([spnet.py L31-32](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L31-L32)) |
| Parameters per type: `a` and `d` 0.02 and 8 (excitatory) or 0.1 and 2 (inhibitory), `b` 0.2, `c` -65 | [spnet.m L9-10][orig-m]; [spnet.cpp L33-37][orig-c] | defaults are the excitatory values ([spnet.py L19-27](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L19-L27)); the runtime test sets the inhibitory ones ([runtime test L95-106](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L95-L106)) | agree when set as in the test |
| Initial state: voltage -65, recovery 0.2 times the voltage | [spnet.m L46-47][orig-m]; [spnet.cpp L91-92][orig-c] | [kernels L164-180](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L164-L180) (voltage `c`, recovery `c` times `b`) | agree for `b` 0.2, `c` -65 |
| Synaptic current: a pulse of the weight on the arrival tick, summed over spikes, cleared every tick; no conductance or synaptic decay in either | [spnet.m L52][orig-m]; [spnet.m L66][orig-m]; [spnet.cpp L111][orig-c]; [spnet.cpp L132][orig-c] | [kernels L633-646](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L633-L646); [kernels L857-859](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L857-L859) | agree |
| External input: one random neuron per ms receives 20 | [spnet.m L53][orig-m]; [spnet.cpp L112-113][orig-c] | the input is a tensor supplied by the caller and added to the current ([kernels L202-222](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L202-L222)); the test feeds a recorded list of such pulses and later one random neuron per tick ([runtime test L284-294](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L284-L294), [runtime test L419-424](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L419-L424)) | cannot compare (the input is not part of the simulator) |
| Delays: excitatory 1 to 20 ms in consecutive blocks of the synapse list, equal in size (C) or random per synapse (MATLAB), inhibitory 1 ms | [spnet.cpp L58-70][orig-c]; [spnet.m L20][orig-m]; [spnet.m L26][orig-m] | the synapses of one source and type are cut into consecutive blocks, one per delay of the range, of nearly equal size ([connections L193-200](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/connections_manager/connections_manager_kernels_logic.proto#L193-L200), [connections L259-266](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/connections_manager/connections_manager_kernels_logic.proto#L259-L266)); the test uses 0 to 19 and fixed 0 ([runtime test L75-94](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L75-L94)) | agree with the C program (numbering shifted by one); the MATLAB program differs |
| Arrival: a spike emitted on tick `t` acts after the delay in ticks | [spnet.m L63-69][orig-m]; [spnet.cpp L127-135][orig-c] | [kernels L586-587](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L586-L587) | agree (same-tick arrival for delay 0) |
| Trace of the most recent spike only (reset, not summed), decay 0.95 per tick | [spnet.m L57][orig-m]; [spnet.m L73][orig-m]; [spnet.cpp L119-120][orig-c]; [spnet.cpp L142-143][orig-c] | tick counters since the last spike ([kernels L401-427](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L401-L427)); last spike tick per neuron ([kernels L383-399](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L383-L399)); table $q\rho^k$ ([spnet.cu L316-345](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet.cu#L316-L345)) | agree |
| Potentiation at a postsynaptic spike from the presynaptic trace delay plus 1 ticks back | [spnet.m L58-60][orig-m]; [spnet.cpp L121][orig-c] | [kernels L939-940](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L939-L940); [kernels L981-983](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L981-L983) | agree on the single-tick path (CPU default). The GPU default batches 32 ticks ([spnet.py L269-273](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L269-L273), [kernels L1166-1267](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L1166-L1267)): same terms, read but not traced tick by tick; the test compares both devices with the same reference weights ([runtime test L392-417](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L392-L417)) |
| Depression at the arrival of a presynaptic spike from the postsynaptic trace, same tick included | [spnet.m L67][orig-m]; [spnet.cpp L133-134][orig-c] | [kernels L664-691](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L664-L691) | agree |
| Amplitudes: 0.1 for potentiation and 0.12 for depression (MATLAB writes it as 1.2 times 0.1) | [spnet.m L57][orig-m]; [spnet.m L67][orig-m]; [spnet.cpp L119-120][orig-c] | learning rate times `ltp_max` and `ltd_max`, class defaults 0.05, 1.0, 1.2 ([spnet.py L28-30](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L28-L30), [spnet.py L74](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L74)); 0.1, 1.0, 1.2 in the test ([runtime test L75-83](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L75-L83)) | agree in the test; the defaults give half the original amplitude |
| Cut-off of the traces | none | table values below `stdp_threshold` (default 1e-5) are dropped, 225 entries for `ltp_max` 1.0 ([spnet.cu L176-182](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet.cu#L176-L182), [spnet.py L107](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L107)) | differ (each dropped value is below 1e-5 before the learning rate) |
| Which synapses learn | excitatory only, fixed in the code | any synapse type with learning rate above 0 ([connections L282](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/connections_manager/connections_manager_kernels_logic.proto#L282), [kernels L1312-1316](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L1312-L1316)) | differ in design; the same in the test (inhibitory rate 0, [runtime test L84-93](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L84-L93)) |
| Weight update: add the constant and the accumulator, clip to 0 and 10, then decay the accumulator | [spnet.m L80-81][orig-m]; [spnet.cpp L168-171][orig-c] | [kernels L1348-1355](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L1348-L1355); [kernels L1355-1360](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L1355-L1360); defaults 0.01, 0.9, 0 and 10 ([spnet.py L77-82](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L77-L82)) | agree |
| When the update runs | once per 1000 ticks, in the main loop | at the end of each `process_ticks` call that has `do_apply_deltas` set, which is the default ([spnet.cu L636-640](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet.cu#L636-L640)); the accumulators persist between calls unless the context is reset ([spnet.cu L632-634](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet.cu#L632-L634)) | differ by design; the test applies once per 1000 ticks, as the original |
| Weights at the start: 6 excitatory, -5 inhibitory | [spnet.m L29][orig-m]; [spnet.cpp L50-51][orig-c] | set per synapse type; in the test [runtime test L75-94](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L75-L94) | agree in the test |
| Network topology: 100 targets per neuron, no self or repeated synapse, inhibitory to excitatory only | [spnet.m L13-27][orig-m]; [spnet.cpp L39-49][orig-c] | built by the synapse growth engine with probabilities and counts; the test loads an explicit edge list ([runtime test L47](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L47)) | cannot compare (Spiky-specific) |
| Arithmetic | single precision (C), double (MATLAB) | single precision with fused multiply-add, or 32-bit fixed point with the integer summation ([kernels L802-853](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L802-L853), [kernels L633-637](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L633-L637)) | cannot compare bitwise |

## Where Spiky goes beyond the original

Per-type STDP tables (each neuron type has its own `ltp_max`, `ltd_max` and `stdp_decay`, taken from the postsynaptic neuron: [spnet.cu L247-345](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet.cu#L247-L345), [kernels L668](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L668), [runtime L940](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime.cu#L940)), per-synapse-type learning parameters, batches of independent networks, the subtractive reset and the refractory window, and the synapse growth engine. None of this is in the original.

## Discrepancies found

No discrepancy in the computation was found at the matched parameters, but nothing was run here, and the batched GPU path of the potentiation was not traced line by line. Two inconsistencies inside Spiky:

- The comment of `SynapseMeta` says the derivative is clipped to the weight bounds ([spnet.py L68-72](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py#L68-L72)); the kernel clips the weight ([kernels L1349-1353](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L1349-L1353)) and not the accumulator.
- The math test kernel `test_math` compares `v > spike_threshold` ([kernels L889](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L889)), the runtime uses `>=` ([kernels L253](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L253), [kernels L785](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime_kernels_logic.proto#L785)); the test kernel is compiled only with profiling enabled ([runtime L765-766](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet/spnet_runtime.cu#L765-L766)).

The reference spike, voltage and weight files of the runtime test are said to match "izhikevitch_model.cpp" ([runtime test L49](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests/test_izhikevitch_runtime.py#L49)); that file is not in the repository, so it is not known how the files were produced.

Back to the [SpNet area](README.md).

[orig-m]: https://web.archive.org/web/20230607050147/http://izhikevich.org/publications/spnet.m
[orig-c]: https://web.archive.org/web/20230607050057/http://izhikevich.org/publications/spnet.cpp
