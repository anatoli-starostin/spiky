"""izhik_sim.py — a general spiking simulator over an arbitrary directed graph.

Two neuron types are supported:

  'izh'  Izhikevich, matched EXACTLY to the spiky-repo native SPNet
         (native/spiky/spnet/spnet_runtime_kernels_logic.proto :: test_math_logic
          and spnet.h :: N_EULER_STEPS=2, EULER_DT=0.5):

             per tick:
               if v > spike_threshold:            # reset from LAST tick's spike
                   v = c ; u += d
               for _ in range(N_EULER_STEPS=2):   # two 0.5 ms half-steps
                   v += EULER_DT*((cf2*v + cf1)*v + cf0 - u + I)
               u += a*(b*v - u)                    # one full u step / tick
               spike this tick iff v >= spike_threshold

         Current enters as +I (the native code uses `cf0 - u + inp`; the
         docstring in spnet.py that says '-I' is a sign-of-inhibition note — the
         COMPILED implementation, the source of truth, uses +inp). Defaults
         a=0.02 b=0.2 c=-65 d=8 cf2=0.04 cf1=5 cf0=140 threshold=30; init v=c,
         u=b*c. I per tick = external injection + sum of synaptic impulses that
         land on the neuron this tick.

  'lif'  Leak-free integrate-and-fire — the native model of OUR exact LUT
         construction (lut_spiking.js simulateCircuit). v integrates a running
         'slope' (ramp synapses) and takes instantaneous 'step' bumps; fires at a
         per-neuron threshold, then latches (fired-once, like simulateCircuit).
         Included so the exact construction can be hosted and validated in this
         same general graph representation. NOT Izhikevich.

Synapses: (src, dst, delay:int ticks, weight, kind). kind for 'izh' targets is
always an impulse into I; for 'lif' targets kind is 'cur' (ramp: slope+=w) or
'imp' (step: v+=w). Inputs are neurons given an explicit fire tick (latency code).
"""

N_EULER_STEPS = 2
EULER_DT = 0.5


class Neuron:
    __slots__ = ("nid", "type", "theta", "a", "b", "c", "d", "cf2", "cf1", "cf0", "vth")

    def __init__(self, nid, ntype="izh", theta=0.5, a=0.02, b=0.2, c=-65.0, d=8.0,
                 cf2=0.04, cf1=5.0, cf0=140.0, vth=30.0):
        self.nid = nid
        self.type = ntype
        self.theta = theta          # lif threshold
        self.a, self.b, self.c, self.d = a, b, c, d
        self.cf2, self.cf1, self.cf0, self.vth = cf2, cf1, cf0, vth


def simulate(neurons, synapses, input_fire, Tmax, record_all=False):
    """neurons: list[Neuron]; synapses: list of (src,dst,delay,weight,kind);
    input_fire: {nid: tick} forced source spikes (latency code);
    returns dict nid -> first spike tick (or None), and optionally all spikes."""
    nmap = {n.nid: n for n in neurons}
    forced_ids = set(input_fire.keys())
    out_syn = {}
    for (src, dst, delay, w, kind) in synapses:
        out_syn.setdefault(src, []).append((dst, int(delay), w, kind))

    # per-neuron dynamic state
    v = {}
    u = {}
    slope = {}
    fired_at = {}          # first spike tick
    latched = {}           # lif fires once then latches
    all_spikes = {}
    for n in neurons:
        if n.type == "izh":
            v[n.nid] = n.c
            u[n.nid] = n.b * n.c
        else:
            v[n.nid] = 0.0
        slope[n.nid] = 0.0
        fired_at[n.nid] = None
        latched[n.nid] = False
        all_spikes[n.nid] = []

    # scheduled synaptic arrivals: arrivals[tick] = list of (dst, w, kind)
    arrivals = {}

    def emit(nid, t):
        if fired_at[nid] is None:
            fired_at[nid] = t
        all_spikes[nid].append(t)
        for (dst, delay, w, kind) in out_syn.get(nid, []):
            arrivals.setdefault(t + delay, []).append((dst, w, kind))

    # forced input spikes (latency-coded sources)
    for nid, t in input_fire.items():
        # forced sources fire exactly once at their scheduled tick
        arrivals.setdefault(t, [])  # ensure the tick is visited
    forced = {}
    for nid, t in input_fire.items():
        forced.setdefault(t, []).append(nid)

    for t in range(0, Tmax + 1):
        # 1) forced source spikes at this tick (skip refs to pruned neurons)
        for nid in forced.get(t, []):
            if nid in nmap and not latched[nid]:
                latched[nid] = True
                emit(nid, t)
        # 2) deliver synaptic arrivals scheduled for this tick
        inj = {}       # per-tick impulse current for izh / step for lif
        touched = set()  # lif neurons that received an arrival this tick (event-driven fire-check)
        for (dst, w, kind) in arrivals.get(t, []):
            if dst not in nmap:
                continue
            n = nmap[dst]
            if n.type == "lif":
                if latched[dst]:
                    continue
                if kind == "cur":
                    slope[dst] += w      # ramp: sustained slope
                else:
                    v[dst] += w          # step: instantaneous bump
                touched.add(dst)
            else:
                inj[dst] = inj.get(dst, 0.0) + w   # izh: impulse into I this tick
        # 3) integrate + threshold
        for n in neurons:
            nid = n.nid
            if nid in forced_ids:
                continue
            if n.type == "lif":
                if latched[nid]:
                    continue
                v[nid] += slope[nid]                # leak-free integrate one tick (keeps V exact)
                # event-driven readout: only CHECK threshold on ticks with an arrival,
                # matching the original event-driven construction — a ramp that drifts
                # past threshold between events must NOT spuriously fire.
                if nid in touched and v[nid] > n.theta:
                    latched[nid] = True
                    emit(nid, t)
            else:
                vv = v[nid]
                uu = u[nid]
                if vv > n.vth:                       # reset from last tick's spike
                    vv = n.c
                    uu += n.d
                I = inj.get(nid, 0.0)
                for _ in range(N_EULER_STEPS):
                    vv = vv + EULER_DT * ((n.cf2 * vv + n.cf1) * vv + n.cf0 - uu + I)
                uu = uu + n.a * (n.b * vv - uu)
                v[nid] = vv
                u[nid] = uu
                if vv >= n.vth:
                    emit(nid, t)

    if record_all:
        return fired_at, all_spikes
    return fired_at


# alias
simulate_graph = simulate
