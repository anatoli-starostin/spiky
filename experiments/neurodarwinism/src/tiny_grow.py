"""exp012 growable nets: a fixed max-capacity genome plus a per-neuron ACTIVE bit.

WHY THIS IS A SEPARATE MODULE AND NOT AN EDIT TO tiny_snn.py. The brief said to extend
tiny_snn.py. Doing that in place would change its geometry from [27, 16] to [67, 56], and
every checkpoint already on disk -- the asexual leader, the crossover leader, the lateral-
inhibition leader -- is a [27, 16] genome. `tiny_final_eval`, `tiny_diag` and `tiny_analyse`
all load those directly, so rebinding the geometry would make all three existing results
unreadable and unreproducible. tiny_snn.py is therefore left exactly as it is, this file
carries the growable substrate, and everything geometry-independent (the episode runner, the
scoring, the target, the metas) is imported from tiny_snn rather than duplicated. Fold it back
in if you would rather -- it is a mechanical move once the old checkpoints are migrated.

CAPACITY. 40 excitatory + 10 inhibitory hidden SLOTS, always allocated in the packed net,
plus the fixed 17 inputs and 6 outputs:

    sources 67 = 17 in + 40 exc + 10 inh          targets 56 = 40 exc + 10 inh + 6 out

An inactive slot is a neuron that exists in the simulation but has no synapses at all, so it
receives nothing and emits nothing. That is what makes a fixed allocation safe: candidates of
different sizes still pack into one SpikingNet with identical per-candidate id blocks.

INVARIANTS, all enforced by one `enforce()` at the end of every genome-producing operation:
  * DALE      sign is a function of the source row -- inputs and exc positive, the 10 inh
              rows negative. A weight may reach 0, never cross it.
  * LEGAL     in->exc only (in->inh, in->out illegal); inh->exc and inh->inh legal but
              inh->out illegal; exc->anything legal.
  * PINNED    every inh->exc delay is exactly 1.
  * ACTIVE    no edge touches an inactive slot, in either direction.
"""
import numpy as np
import torch

import tiny_snn as S
from tiny_snn import (D_HI, D_LO, N_IN, N_METAS, N_OUT, N_TICKS, READOUT_WINDOW,  # noqa: F401
                      affine_ceiling_and_r, constant_baseline, decode_actions,
                      fit_target_stats, jsonable, kendall_tau_b, metas, target_offsets)

# ------------------------------------------------------------------ geometry
# OUT_PER_TARGET output neurons per TARGET DIMENSION. Each target reads from its own dedicated
# group and from no other, so the readout stays strictly NON-MIXING: a fixed aggregate of the
# group's first-spike times, then one evolved scale and shift.
#
# The point is to give the NETWORK more machinery to disentangle with, rather than giving the
# readout the power to unmix. K = 1 reproduces the historical 6-output substrate exactly.
N_TARGET = N_OUT                                # 6 target dimensions, fixed by the task
OUT_PER_TARGET = 1                              # K; set via set_out_per_target()
N_OUT_NEURONS = N_TARGET * OUT_PER_TARGET
OUT_AGG = "mean"                                # "mean" | "min" -- fixed, never evolved

N_EXC_MAX, N_INH_MAX = 40, 10
N_SRC = N_IN + N_EXC_MAX + N_INH_MAX            # 67
N_TGT = N_EXC_MAX + N_INH_MAX + N_OUT_NEURONS
R_IN = slice(0, N_IN)                           # 0..16
R_EXC = slice(N_IN, N_IN + N_EXC_MAX)           # 17..56
R_INH = slice(N_IN + N_EXC_MAX, N_SRC)          # 57..66
C_EXC = slice(0, N_EXC_MAX)                     # 0..39
C_INH = slice(N_EXC_MAX, N_EXC_MAX + N_INH_MAX)  # 40..49
C_OUT = slice(N_EXC_MAX + N_INH_MAX, N_TGT)

def set_hidden_capacity(n_exc, n_inh):
    """Resize the hidden layer ITSELF, not just how many slots start active.

    --init-exc/--init-inh only choose how many of the 40+10 slots are switched on; build()
    still allocates all fifty, and the inactive ones sit in the engine as silent neurons. For
    a run whose whole point is a radically smaller network that is misleading -- the reported
    topology would not be the built one -- so this shrinks the capacity for real.

    n_inh = 0 removes the inhibitory rows and columns entirely: no inh->exc pin, no negative
    Dale half, and inh_coeff becomes meaningless (do not enable it).

    Call before any genome exists, and before set_target_dims/set_out_per_target, which
    rebuild the geometry that depends on these.
    """
    global N_EXC_MAX, N_INH_MAX, N_SRC, N_TGT, R_IN, R_EXC, R_INH, SIGN
    N_EXC_MAX, N_INH_MAX = int(n_exc), int(n_inh)
    N_SRC = N_IN + N_EXC_MAX + N_INH_MAX
    R_IN = slice(0, N_IN)
    R_EXC = slice(N_IN, N_IN + N_EXC_MAX)
    R_INH = slice(N_IN + N_EXC_MAX, N_SRC)
    SIGN = np.ones(N_SRC, np.float64)
    SIGN[R_INH] = -1.0
    set_out_per_target(OUT_PER_TARGET, OUT_AGG)     # rebuilds N_TGT, the slices, LEGAL, pins
    return N_SRC, N_TGT


# ---- RESTRICTING THE TASK TO A SUBSET OF THE SIX TARGET DIMENSIONS.
# None means all six, which is the historical behaviour and the default: with TARGET_DIMS
# None, N_TARGET == N_OUT == 6 and every expression below is the one that was there before,
# so every existing checkpoint and run reproduces bit for bit.
#
# A single dimension is a CAPACITY test: it removes the 6-way output split, all cross-target
# mixing, and the per-dimension misalignment in one move, and hands the entire output budget
# to one number. Note the baseline moves with it -- a single dimension must be judged against
# ITS OWN constant-predictor MSE, not against the 6-dim 34.15.
TARGET_DIMS = None


def set_target_dims(dims):
    """Restrict the task to `dims` (indices into the six). None restores all six.

    Rebuilds the geometry, so it must be called BEFORE any genome exists -- and before or
    together with set_out_per_target(), whose output-layer size is N_TARGET * K.
    """
    global TARGET_DIMS, N_TARGET
    if dims is None:
        TARGET_DIMS, N_TARGET = None, N_OUT
    else:
        TARGET_DIMS = [int(d) for d in dims]
        assert all(0 <= d < N_OUT for d in TARGET_DIMS), "target dim out of range"
        assert len(set(TARGET_DIMS)) == len(TARGET_DIMS), "duplicate target dim"
        N_TARGET = len(TARGET_DIMS)
    return set_out_per_target(OUT_PER_TARGET, OUT_AGG)


# ---- LUT TARGET / LUT DECODE.  Default None = every earlier run is byte-identical.
#
# The offset target asks the net to hit a centred, 2.5-sigma-clipped, uniformly quantised
# level. A first-spike readout can emit exactly 33 things (ticks 0..31, plus silence), so the
# natural target is a 32-entry LOOKUP TABLE indexed by that spike time -- which the readout
# can represent exactly rather than approximate. LUT[b] is the mean of the continuous target
# over bin b, the MSE-optimal decode for that bin.
#
# Index 32 is silence. It decodes to the training mean of y', i.e. the best constant: "this
# output never fired" carries no information about which bin the sample is in, so guessing
# the mean is the honest decode rather than pinning it to the last bin.
LUT_TABLE = None       # length 33, in the units of the raw action
LUT_EDGES = None       # 31 interior quantile edges, on the CONTINUOUS target
LUT_DIM = None


def set_lut_task(edges, table, dim):
    """Install the LUT target and its decode. None clears it back to the offset target."""
    global LUT_TABLE, LUT_EDGES, LUT_DIM
    if edges is None:
        LUT_TABLE, LUT_EDGES, LUT_DIM = None, None, None
        return None
    LUT_EDGES = np.asarray(edges, np.float64)
    LUT_TABLE = np.asarray(table, np.float64)
    LUT_DIM = int(dim)
    assert LUT_TABLE.shape == (READOUT_WINDOW + 1,), \
        f"LUT must be {READOUT_WINDOW} decode values plus one for silence"
    assert LUT_EDGES.shape == (READOUT_WINDOW - 1,), "need one interior edge between each bin"
    # A LUT target IS one dimension, so bind the task to it here rather than relying on the
    # caller to also pass --target-dims. Without this N_TARGET stays 6, the output layer is
    # built six neurons wide, and score() silently broadcasts one target across all six --
    # finite, plausible, and not the network anyone asked for.
    set_target_dims([LUT_DIM])
    return LUT_TABLE


def lut_bins(Y):
    """The bin index 0..31 of each sample under the installed LUT."""
    return np.digitize(np.asarray(Y)[:, LUT_DIM], LUT_EDGES)


# ---- SIGN-COMPARISON BIT TARGET.  Default None = every earlier run is byte-identical.
#
# The teacher's action depends on its input ONLY through 192 bits of the form
# sign(x_norm[a] - x_norm[b]), then a table lookup. Under the latency encoder -- larger x
# fires EARLIER -- such a bit is exactly "which of these two input spikes arrives first",
# which is the one computation a delay-based spiking net should be native at. So this is the
# atomic probe of the whole approach.
#
# Note this target is a function of X, not of Y, which is why task_targets takes X.
BIT_TASK = None            # (a, b) input indices, or None

# ---- LIF NEURONS.  None = the Izhikevich metas, i.e. every earlier run byte-identical.
# A dict of LIFNeuronMeta kwargs (tau, threshold, v_rest, v_reset, refractory_ticks,
# subtractive_reset) switches all four neuron metas to leaky integrate-and-fire.
LIF = None


def set_lif(**kw):
    """Install LIF neuron metas, or clear back to Izhikevich with set_lif(tau=None)."""
    global LIF
    LIF = None if kw.get("tau") is None else dict(kw)
    return LIF


# ---- WIDE EPISODES.  All None = the historical 32-tick input / 96-tick episode / 64 delays,
# so every earlier run is byte-identical. A wider input window spreads the latency code over
# more ticks, which is what gives a spike-order circuit a usable timing gap between inputs.
T_IN_OVR = None
N_TICKS_OVR = None
RO_WIN_OVR = None
D_HI_OVR = None


def set_episode(t_in=None, n_ticks=None, readout_window=None, d_hi=None):
    global T_IN_OVR, N_TICKS_OVR, RO_WIN_OVR, D_HI_OVR
    T_IN_OVR, N_TICKS_OVR, RO_WIN_OVR, D_HI_OVR = t_in, n_ticks, readout_window, d_hi
    if d_hi is not None:
        DLY_HI[:] = d_hi
    return dict(t_in=t_in_(), n_ticks=n_ticks_(), readout_window=ro_win_(), d_hi=d_hi_())


def t_in_():
    return S.T_IN if T_IN_OVR is None else int(T_IN_OVR)


def n_ticks_():
    return N_TICKS if N_TICKS_OVR is None else int(N_TICKS_OVR)


def ro_win_():
    return READOUT_WINDOW if RO_WIN_OVR is None else int(RO_WIN_OVR)


def d_hi_():
    return D_HI if D_HI_OVR is None else int(D_HI_OVR)


def metas_(w_ceiling):
    """tiny_snn.metas, but spanning [D_LO, d_hi_()] so delays past 64 have a meta to land on.
    Without this a delay > 64 indexes off the end of the bank and the build dies inside
    add_connections with a cudaErrorMemoryAllocation that looks like GPU exhaustion."""
    if D_HI_OVR is None:
        return metas(w_ceiling)
    from spiky.spnet.spnet import SynapseMeta
    return [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                        min_weight=-w_ceiling, max_weight=w_ceiling, initial_noise_level=0.0,
                        weight_decay=0.9, weight_scaling_cf=0.0,
                        _forward_group_size=S.GROUP_SIZE, _backward_group_size=S.GROUP_SIZE)
            for d in range(D_LO, d_hi_() + 1)]


def set_bit_task(ia, ib):
    """Score against 1[x_norm[ia] > x_norm[ib]]. None clears it."""
    global BIT_TASK
    if ia is None:
        BIT_TASK = None
        return None
    assert 0 <= int(ia) < N_IN and 0 <= int(ib) < N_IN and int(ia) != int(ib)
    BIT_TASK = (int(ia), int(ib))
    set_target_dims([0])       # one output; TARGET_DIMS is unused while BIT_TASK is set
    return BIT_TASK


def task_targets(Y, X=None):
    """The target this run is actually being scored on -> [B, N_TARGET]."""
    if BIT_TASK is not None:
        assert X is not None, "the bit task is a function of X, so X must be passed"
        a_, b_ = BIT_TASK
        return (np.asarray(X)[:, a_] > np.asarray(X)[:, b_]).astype(np.float64)[:, None]
    if LUT_TABLE is not None:
        return LUT_TABLE[lut_bins(Y)][:, None]
    t = target_offsets(Y)
    return t if TARGET_DIMS is None else t[:, TARGET_DIMS]


def task_baseline(Y, X=None):
    """Best-constant-predictor MSE on the dimensions this run is scored on."""
    t = task_targets(Y, X)
    return float(((t - t.mean(0)) ** 2).mean())


def set_out_per_target(k, agg="mean"):
    """Rebuild every geometry-dependent constant for K output neurons per target.

    Called before any genome exists. Touches the module globals rather than threading a shape
    through fifty call sites -- the alternative is a constructor-based rewrite of the whole
    module, which is not worth it for a substrate knob that changes once per run.
    """
    global OUT_PER_TARGET, N_OUT_NEURONS, N_TGT, C_EXC, C_INH, C_OUT, OUT_AGG
    global LEGAL, PIN_DELAY, DLY_LO, DLY_HI
    OUT_PER_TARGET = int(k)
    OUT_AGG = agg
    N_OUT_NEURONS = N_TARGET * OUT_PER_TARGET
    N_TGT = N_EXC_MAX + N_INH_MAX + N_OUT_NEURONS
    C_EXC = slice(0, N_EXC_MAX)
    C_INH = slice(N_EXC_MAX, N_EXC_MAX + N_INH_MAX)
    C_OUT = slice(N_EXC_MAX + N_INH_MAX, N_TGT)
    LEGAL = np.ones((N_SRC, N_TGT), bool)
    LEGAL[R_IN, C_INH] = False
    LEGAL[R_IN, C_OUT] = False
    LEGAL[R_INH, C_OUT] = False
    PIN_DELAY = np.zeros((N_SRC, N_TGT), bool)
    PIN_DELAY[R_INH, C_EXC] = True
    DLY_LO = np.full(N_TGT, D_LO, np.int64)
    DLY_HI = np.full(N_TGT, D_HI, np.int64)
    return N_OUT_NEURONS


SIGN = np.ones(N_SRC, np.float64)
SIGN[R_INH] = -1.0

LEGAL = np.ones((N_SRC, N_TGT), bool)
LEGAL[R_IN, C_INH] = False
LEGAL[R_IN, C_OUT] = False
LEGAL[R_INH, C_OUT] = False

PIN_DELAY = np.zeros((N_SRC, N_TGT), bool)
PIN_DELAY[R_INH, C_EXC] = True
PIN_VALUE = 1

DLY_LO = np.full(N_TGT, D_LO, np.int64)
DLY_HI = np.full(N_TGT, D_HI, np.int64)

E_TARGET = 0.80            # the excitatory fraction grow/shrink steer towards
MIN_EXC = 2                # shrink will not take the excitatory pool below this

# ---- NORMALISED WEIGHTS AND A CONSTANT BUILD-TIME GAIN.
# The genome now stores MAGNITUDES IN [0, 1] (excitatory rows) and [-1, 0] (inhibitory rows).
# build() multiplies by GAIN once, on the way into the engine; nothing else -- readout,
# episode runner, scoring -- changes at all.
#
# THIS IS A PURE REPARAMETERISATION AND IS MEANT TO STAY ONE. GAIN is set to the old Dale
# ceiling, so normalised 1.0 maps to exactly the absolute 200.0 that w_ceiling used to allow,
# and every mutation constant below is the old ABSOLUTE constant divided by GAIN rather than
# a fresh number. Written as the division on purpose: if GAIN ever moves, the mutation scale
# moves with it and the operators keep their absolute meaning.
GAIN = 200.0
W_SIGMA_NORM = 9.0 / GAIN                      # old w_sigma 0.15 * w_max 60 = 9.0 absolute
BORN_LO, BORN_HI = 6.0 / GAIN, 36.0 / GAIN     # old uniform(0.10, 0.60) * 60 = 6..36
W_INIT_NORM = 60.0 / GAIN                      # old random_genome drew uniform(0, w_max=60)


# ------------------------------------------------------------------ active mask
def active_rows(g):
    a = np.ones(N_SRC, bool)
    a[R_EXC] = g["act_exc"]
    a[R_INH] = g["act_inh"]
    return a


def active_cols(g):
    a = np.ones(N_TGT, bool)
    a[C_EXC] = g["act_exc"]
    a[C_INH] = g["act_inh"]
    return a


def active_cells(g):
    return active_rows(g)[:, None] & active_cols(g)[None, :]


def n_active(g):
    return int(g["act_exc"].sum() + g["act_inh"].sum())


def e_fraction(g):
    n = n_active(g)
    return float(g["act_exc"].sum()) / n if n else 0.0


def enforce(g):
    """Re-impose every structural invariant, in place. -> the same genome.

    When QUANTIZED, the grid is one of those invariants: snapping happens here, so every
    genome-producing path (init, mutate, crossover, grow, the seed embedding) lands on-grid
    without any of them needing to know about it.
    """
    g["mask"] &= LEGAL & active_cells(g)
    if DELAY_LEVELS is not None:
        g["delay"] = snap_delays(g["delay"])
    g["delay"][PIN_DELAY] = PIN_VALUE          # AFTER the snap, so the pin always wins
    if QUANTIZED:
        g["weight"] = snap(g["weight"])
    return g


# ------------------------------------------------------------------ genome
# ---- EVOLVABLE READOUT CALIBRATION.
# Twelve extra genes: a per-output gain a_j and offset b_j, so the readout becomes
#     y_j = a_j * raw_j + b_j
# with raw_j the existing TTFS first-spike offset. a=1, b=0 is exactly the old behaviour, and
# a genome without the field defaults to that -- so every checkpoint written before this
# change still reproduces its own MSE bit for bit.
#
# WHY THIS IS WORTH TWELVE GENES. Every net in exp012 has lost 6-9 MSE to `scale error`: the
# first spike is a MIN over afferents, a min compresses variance, and the readout cannot span
# the target's range however good the network is. The affine ceiling has been reported all
# along as the diagnostic for exactly that gap. These genes let evolution actually claim it.
# ---- OPTIONAL WEIGHT QUANTISATION.
# 11 symmetric levels at 0.2 spacing. Dale splits the grid rather than being re-checked after
# the fact: an excitatory source row may only take a level from the NON-NEGATIVE half, an
# inhibitory row only from the NON-POSITIVE half, so quantisation and Dale's law are the same
# operation and a snapped weight cannot violate either.
#
# Off by default. When off, every weight path below behaves exactly as before -- continuous
# magnitudes with a Gaussian step -- so existing runs and checkpoints are untouched.
QUANT_LEVELS = np.round(np.arange(-1.0, 1.0001, 0.2), 10)          # -1.0 .. 1.0, 11 levels
QUANT_POS = QUANT_LEVELS[QUANT_LEVELS >= 0]                        # 0.0 .. 1.0, 6 levels
QUANT_NEG = QUANT_LEVELS[QUANT_LEVELS <= 0]                        # -1.0 .. 0.0, 6 levels
QUANT_STEP = 0.2                   # only meaningful for an evenly-spaced grid
QUANTIZED = False                  # module-level default; the driver flips it per run

# ---- OPTIONAL DELAY GRID, independent of the weight grid.
# None = a delay is a free integer in [D_LO, D_HI], the historical behaviour.
DELAY_LEVELS = None

# ---- OPTIONAL HARD FAN-OUT CAP on HIDDEN neurons (the recurrent exc + inh units).
# A ceiling on outgoing synapses per hidden row, counting all of its targets -- exc, inh and
# output alike. Inputs are NOT capped (they are already restricted to exc-only), and outputs
# have no outgoing edges at all.
#
# ENFORCED BY REFUSAL, NOT BY REPAIR. The add and grow operators simply decline to propose an
# edge that would breach the cap; nothing trims afterwards. That distinction matters: a
# repair pass would silently delete whatever a legitimate operator had just decided to add,
# so the search would be optimising against an edit it cannot see. Crossover needs no special
# handling -- a hidden slot inherits its WHOLE outgoing row from one parent, so if both
# parents are under the cap the child is too, and enforce() only ever removes edges.
#
# None = off, which is the default, so continuous runs are byte-for-byte unchanged.
FANOUT_CAP = None
HID_LO = N_IN                      # rows HID_LO.. are the hidden units (exc then inh)


def hidden_fanout(g):
    return g["mask"][HID_LO:].sum(1)


def fanout_ok(g, cap=None):
    """No hidden neuron exceeds the cap. -> (ok, n_violating_neurons)."""
    c = FANOUT_CAP if cap is None else cap
    if c is None:
        return True, 0
    bad = int((hidden_fanout(g) > c).sum())
    return bad == 0, bad


def _trim_hidden_fanout(g, rng):
    """Init-time only: subsample any hidden row that came out over the cap."""
    if FANOUT_CAP is None:
        return g
    for i in np.where(hidden_fanout(g) > FANOUT_CAP)[0]:
        row = HID_LO + int(i)
        js = np.where(g["mask"][row])[0]
        drop = rng.choice(js, size=len(js) - FANOUT_CAP, replace=False)
        g["mask"][row, drop] = False
    return g


def _headroom(g):
    """Remaining out-edge budget per SOURCE row; inf for rows the cap does not apply to."""
    h = np.full(N_SRC, np.inf)
    if FANOUT_CAP is not None:
        h[HID_LO:] = FANOUT_CAP - hidden_fanout(g)
    return h


def _limit_adds(add, g, rng):
    """Drop proposed adds that would push a hidden row over the cap. Refusal, not repair."""
    if FANOUT_CAP is None:
        return add
    room = _headroom(g)
    for i in np.where(add.any(1))[0]:
        r = room[i]
        n = int(add[i].sum())
        if n > r:
            js = np.where(add[i])[0]
            keep = rng.choice(js, size=max(int(r), 0), replace=False) if r > 0 else []
            add[i] = False
            add[i, keep] = True
    return add


def set_weight_levels(levels):
    """Install an arbitrary weight grid and re-derive the two Dale halves.

    The grid need not be evenly spaced or symmetric. {-0.5, 0, 1.0} gives excitatory rows
    {0, 1.0} and inhibitory rows {-0.5, 0} -- binary within each sign class, with different
    E and I magnitudes. Everything downstream indexes the SUB-GRID ARRAYS rather than
    assuming a spacing, so an irregular grid needs no special handling anywhere.
    """
    global QUANT_LEVELS, QUANT_POS, QUANT_NEG, QUANT_STEP
    QUANT_LEVELS = np.array(sorted(float(x) for x in levels), np.float64)
    QUANT_POS = QUANT_LEVELS[QUANT_LEVELS >= 0]
    QUANT_NEG = QUANT_LEVELS[QUANT_LEVELS <= 0]
    d = np.diff(QUANT_LEVELS)
    QUANT_STEP = float(d.min()) if len(d) else 0.0
    assert len(QUANT_POS) >= 1 and len(QUANT_NEG) >= 1, "grid needs a level in each Dale half"
    return QUANT_LEVELS


def set_delay_levels(levels):
    """Install a delay grid; None restores free integers in [D_LO, D_HI]."""
    global DELAY_LEVELS
    if levels is None:
        DELAY_LEVELS = None
    else:
        DELAY_LEVELS = np.array(sorted(int(x) for x in levels), np.int64)
        assert DELAY_LEVELS[0] >= D_LO and DELAY_LEVELS[-1] <= D_HI, "delay grid out of range"
        # if the pinned value were not itself a level, enforce() and snap_delays() would
        # fight over every inh->exc synapse
        assert PIN_VALUE in DELAY_LEVELS, (
            f"the pinned inh->exc delay {PIN_VALUE} must be a legal delay level")
    return DELAY_LEVELS


def _nearest(vals, grid):
    """Nearest grid member, elementwise. Works for any grid, evenly spaced or not."""
    return grid[np.abs(np.asarray(vals)[..., None] - grid).argmin(-1)]


def _hop(vals, grid, steps):
    """Move each value `steps` positions along `grid`, clamped to its ends.

    Index-based, not arithmetic, so it is correct for an irregular grid. On a 2-level
    sub-grid any non-zero step is a toggle-or-stay: from the bottom a +step flips to the top
    and a -step clamps in place, so exactly half of the proposed hops are no-ops. That is
    the intended clamped behaviour, not a bug -- but it does halve the EFFECTIVE weight
    mutation rate on a binary grid.
    """
    idx = np.abs(np.asarray(vals)[..., None] - grid).argmin(-1)
    return grid[np.clip(idx + steps, 0, len(grid) - 1)]


def snap(w):
    """Snap a whole [N_SRC, N_TGT] weight matrix onto each row's legal Dale sub-grid."""
    npos = N_IN + N_EXC_MAX
    out = np.empty_like(w)
    out[:npos] = _nearest(w[:npos], QUANT_POS)
    out[npos:] = _nearest(w[npos:], QUANT_NEG)
    return out


def snap_delays(d):
    return d if DELAY_LEVELS is None else _nearest(d, DELAY_LEVELS).astype(np.int64)


def on_grid(g, tol=1e-6):
    """Every weight sits on a legal level for its Dale half. -> (ok, n_violations).

    Compared with a tolerance rather than exactly: checkpoints store weights as float32, so a
    reloaded 0.2 comes back as 0.200000003 and an exact test would report every single weight
    as off-grid. 1e-6 is six orders of magnitude below the 0.2 spacing, so it cannot mask a
    genuine violation -- the smallest real one would be a half-step, 0.1.
    """
    npos = N_IN + N_EXC_MAX
    w = g["weight"]
    d_pos = np.abs(w[:npos][..., None] - QUANT_POS).min(-1)
    d_neg = np.abs(w[npos:][..., None] - QUANT_NEG).min(-1)
    bad = int((d_pos > tol).sum() + (d_neg > tol).sum())
    return bad == 0, bad


# ---- EVOLVABLE GLOBAL INHIBITION COEFFICIENT.
# One continuous scalar per genome, NOT quantised. It enters in exactly one place: build()
# multiplies every INHIBITORY synaptic current by it, so the effective current is
#     exc:  GAIN * w            inh:  GAIN * w * inh_coeff
# The genome keeps symmetric weights and the asymmetry lives in this one number, which means
# the E/I balance becomes a single evolvable dial rather than something spread across every
# inhibitory synapse.
#
# TWO DIFFERENT DEFAULTS, on purpose. A NEW genome starts at INH_COEFF_INIT = 0.5. A genome
# that has no such gene at all -- every checkpoint written before this change -- reads as 1.0,
# because 1.0 is the value that makes build() reproduce exactly what those checkpoints used
# to do. Starting new genomes at 1.0 instead would lose the intended starting point; defaulting
# old ones to 0.5 would silently halve every inhibitory current in every prior result.
# ---- EVOLVABLE GLOBAL GAIN.
# The constant GAIN becomes a per-genome scalar. It multiplies SYNAPTIC weights only:
#     exc current = gain * w            inh current = gain * w * inh_coeff
# The fixed input injection (200.0, applied to the 17 input neurons by the episode runner) is
# NOT scaled by it. That asymmetry is the only reason the gene means anything: if gain scaled
# the injection too it would be a pure units change and could not move a single spike time.
GAIN_INIT = GAIN                   # 200.0, so a fresh genome starts where the constant was
GAIN_LO, GAIN_HI = 20.0, 2000.0
GAIN_SIGMA = 20.0

INH_COEFF_INIT = 0.5
INH_COEFF_LEGACY = 1.0             # what a genome WITHOUT the gene means
INH_COEFF_LO, INH_COEFF_HI = 0.05, 4.0
INH_COEFF_SIGMA = 0.1

AFF_A_INIT, AFF_B_INIT = 1.0, 0.0
AFF_A_LIM, AFF_B_LIM = 4.0, 64.0     # sane finite box; sign of a is NOT constrained
SIGMA_A = 0.05                       # dimensionless, a lives around 1
SIGMA_B = 0.05 * 6.16                # = 0.308; b is in target units, whose per-dim std is 6.16


def blank(n_exc=8, n_inh=2):
    return dict(mask=np.zeros((N_SRC, N_TGT), bool),
                delay=np.ones((N_SRC, N_TGT), np.int64),
                weight=np.zeros((N_SRC, N_TGT), np.float64),
                act_exc=np.arange(N_EXC_MAX) < n_exc,
                act_inh=np.arange(N_INH_MAX) < n_inh,
                aff_a=np.full(N_TARGET, AFF_A_INIT),
                aff_b=np.full(N_TARGET, AFF_B_INIT),
                inh_coeff=float(INH_COEFF_INIT),
                gain=float(GAIN_INIT))


def gain_of(g):
    """The genome's synaptic gain, defaulting to the historical constant for old genomes."""
    v = g.get("gain")
    return float(GAIN) if v is None else float(v)


def inh_coeff_of(g):
    """The genome's inhibition coefficient, or the legacy 1.0 if it has no such gene."""
    c = g.get("inh_coeff")
    return INH_COEFF_LEGACY if c is None else float(c)


def affine_of(g):
    """(a, b) for a genome, defaulting to the identity for pre-affine genomes."""
    a = g.get("aff_a")
    b = g.get("aff_b")
    return (np.full(N_TARGET, AFF_A_INIT) if a is None else a,
            np.full(N_TARGET, AFF_B_INIT) if b is None else b)


def with_identity_affine(g):
    """A genome guaranteed to carry affine genes, without disturbing one that already does."""
    if "aff_a" in g and "aff_b" in g:
        return g
    h = copy_genome(g)
    h["aff_a"], h["aff_b"] = affine_of(g)
    return h


def random_genome(rng, w_init=W_INIT_NORM, p_init=0.5, n_exc=8, n_inh=2):
    """Random genome. Under quantisation the weight draw is taken FROM THE GRID's own span.

    WHY, and it is not cosmetic. Continuous-then-snap is only valid when the draw range
    actually spans the grid. `w_init` is 0.3, so on the {-1, 0, 1} grid *every* weight rounds
    to the nearest level -- which is 0 -- and the whole initial population is a dead network
    with no synaptic current at all. Measured: 0 of 290 weights non-zero on the symmetric
    grid, 10 of 299 on {-0.5, 0, 1.0}. The 0.2-spaced 11-level grid escapes it only by
    accident, and even there init can never reach past level 0.2.
    So when QUANTIZED the draw is uniform over [0, max level] for the row's own Dale half,
    which snaps to a spread across all levels for any grid.
    """
    g = blank(n_exc, n_inh)
    g["mask"] = (rng.random((N_SRC, N_TGT)) < p_init)
    g["delay"] = rng.integers(DLY_LO[None, :], DLY_HI[None, :] + 1,
                              (N_SRC, N_TGT)).astype(np.int64)
    if QUANTIZED:
        npos = N_IN + N_EXC_MAX
        w = np.empty((N_SRC, N_TGT))
        w[:npos] = rng.uniform(0.0, QUANT_POS.max(), (npos, N_TGT))
        w[npos:] = -rng.uniform(0.0, abs(QUANT_NEG.min()), (N_SRC - npos, N_TGT))
        g["weight"] = w
    else:
        g["weight"] = rng.uniform(0.0, w_init, (N_SRC, N_TGT)) * SIGN[:, None]
    enforce(g)
    return _trim_hidden_fanout(g, rng)


def _born_magnitudes(rng, n, sign):
    """Weight for a newly switched-on edge, one per element of `sign` (+1 exc, -1 inh).

    Under quantisation a new edge is born on a NON-ZERO level: born-at-zero would make
    "add an edge" a no-op, and on a binary grid the old uniform(0.03, 0.18) draw snaps to
    exactly zero every time.
    """
    if not QUANTIZED:
        return rng.uniform(BORN_LO, BORN_HI, n) * sign
    pos = QUANT_POS[QUANT_POS > 0]
    neg = QUANT_NEG[QUANT_NEG < 0]
    out = np.empty(n)
    is_e = np.asarray(sign) > 0
    if is_e.any():
        out[is_e] = rng.choice(pos, int(is_e.sum())) if len(pos) else 0.0
    if (~is_e).any():
        out[~is_e] = rng.choice(neg, int((~is_e).sum())) if len(neg) else 0.0
    return out


def normalize_abs(g_abs, gain=GAIN):
    """An ABSOLUTE-weight genome -> a normalised one. Load-time helper for old checkpoints.

    Non-destructive: returns a new genome, the checkpoint on disk is untouched.
    """
    g = copy_genome(g_abs)
    g["weight"] = g_abs["weight"] / gain
    return g


def effective_weights(g, gain=GAIN):
    """What build() actually hands the engine."""
    return g["weight"] * gain


def copy_genome(g):
    # inh_coeff is a plain float, not an array -- floats are immutable so there is nothing
    # to copy, but they also have no .copy()
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in g.items()}


def _wire_new_neuron(g, rng, row, col, n_in_e=(1, 3), n_out_e=(1, 3)):
    """Give a freshly activated slot a few legal edges among the ACTIVE neurons."""
    ar, ac = active_rows(g), active_cols(g)
    room = _headroom(g)
    # incoming: any active source that may legally reach this column AND has budget left
    src = np.where(ar & LEGAL[:, col] & (room > 0))[0]
    src = src[src != row]
    if len(src):
        k = int(rng.integers(n_in_e[0], n_in_e[1] + 1))
        for i in rng.choice(src, size=min(k, len(src)), replace=False):
            g["mask"][i, col] = True
            g["weight"][i, col] = _born_magnitudes(rng, 1, [SIGN[i]])[0]
            g["delay"][i, col] = rng.integers(D_LO, D_HI + 1)
    # outgoing: any active target this row may legally reach, up to the new unit's own budget
    tgt = np.where(ac & LEGAL[row])[0]
    tgt = tgt[tgt != col]
    if FANOUT_CAP is not None:
        n_out_e = (min(n_out_e[0], FANOUT_CAP), min(n_out_e[1], FANOUT_CAP))
    if len(tgt) and n_out_e[1] > 0:
        k = int(rng.integers(n_out_e[0], n_out_e[1] + 1))
        for j in rng.choice(tgt, size=min(k, len(tgt)), replace=False):
            g["mask"][row, j] = True
            g["weight"][row, j] = _born_magnitudes(rng, 1, [SIGN[row]])[0]
            g["delay"][row, j] = rng.integers(D_LO, D_HI + 1)
    return g


def grow(g, rng):
    """Activate one inactive hidden slot, choosing the type that lands E/(E+I) nearest 0.80.

    -> (genome, what) where what is 'exc', 'inh' or None (no-op at capacity).
    """
    ne, ni = int(g["act_exc"].sum()), int(g["act_inh"].sum())
    room_e, room_i = ne < N_EXC_MAX, ni < N_INH_MAX
    if not (room_e or room_i):
        return g, None
    n = ne + ni + 1
    d_e = abs((ne + 1) / n - E_TARGET) if room_e else np.inf
    d_i = abs(ne / n - E_TARGET) if room_i else np.inf
    kind = "exc" if d_e <= d_i else "inh"          # tie -> exc, as specified
    if kind == "exc":
        s = int(np.argmin(g["act_exc"]))           # lowest-index inactive slot
        g["act_exc"][s] = True
        row, col = N_IN + s, s
    else:
        s = int(np.argmin(g["act_inh"]))
        g["act_inh"][s] = True
        row, col = N_IN + N_EXC_MAX + s, N_EXC_MAX + s
    _wire_new_neuron(g, rng, row, col)
    # enforce() here is not belt-and-braces: _wire_new_neuron draws delays uniform in [1,64],
    # and a new neuron's edges can land on inh->exc cells in either direction (an inhibitory
    # newcomer projecting to exc, or an existing inhibitory unit projecting into a new exc).
    # Without this the pin is violated -- which is exactly what the pre-flight caught.
    return enforce(g), kind


def shrink(g, rng):
    """Deactivate one active hidden neuron and mask all of its edges.

    CHOICE OF VICTIM, stated because the brief left it open: restrict to the LEAST-CONNECTED
    candidates (minimum total degree, in + out), then among those take the one whose removal
    leaves E/(E+I) closest to 0.80. Least-connected first because removing a hub is a large,
    mostly-destructive perturbation, whereas removing a barely-wired neuron is the gentlest
    edit that still reduces size -- and the E-fraction tie-break keeps the pool balanced.

    -> (genome, what) where what is 'exc', 'inh' or None (no-op at the floor).
    """
    ne, ni = int(g["act_exc"].sum()), int(g["act_inh"].sum())
    cand = []                                       # (degree, |e_frac_after - 0.8|, kind, slot)
    deg_out = g["mask"].sum(1)
    deg_in = g["mask"].sum(0)
    n_after = ne + ni - 1
    if n_after < 1:
        return g, None
    for s in np.where(g["act_exc"])[0]:
        if ne - 1 < MIN_EXC:
            break
        d = int(deg_out[N_IN + s] + deg_in[s])
        cand.append((d, abs((ne - 1) / n_after - E_TARGET), "exc", int(s)))
    for s in np.where(g["act_inh"])[0]:
        d = int(deg_out[N_IN + N_EXC_MAX + s] + deg_in[N_EXC_MAX + s])
        cand.append((d, abs(ne / n_after - E_TARGET), "inh", int(s)))
    if not cand:
        return g, None
    dmin = min(c[0] for c in cand)
    pick = min([c for c in cand if c[0] == dmin], key=lambda c: c[1])
    _d, _f, kind, s = pick
    if kind == "exc":
        g["act_exc"][s] = False
    else:
        g["act_inh"][s] = False
    return enforce(g), kind


def mutate(g, rng, p_add=0.02, p_prune=0.02, p_delay=0.08, p_weight=0.25,
           sigma=W_SIGMA_NORM, p_grow=0.03, p_shrink=0.03, p_affine=0.25,
           p_inhcoeff=0.0, p_gain=0.0):
    """Edge mutation exactly as in tiny_snn, plus grow and shrink.

    Weights are normalised, so the step sigma and the born magnitude are the OLD ABSOLUTE
    constants divided by GAIN, and the Dale clip is to the unit half-line rather than to
    +-w_ceiling. In effective units nothing about the operator has changed.
    """
    h = copy_genome(g)

    if rng.random() < p_grow:
        grow(h, rng)
    if rng.random() < p_shrink:
        shrink(h, rng)

    allowed = LEGAL & active_cells(h)
    off = (~h["mask"]) & allowed
    add = _limit_adds(off & (rng.random((N_SRC, N_TGT)) < p_add), h, rng)
    if add.any():
        n = int(add.sum())
        h["mask"][add] = True
        h["weight"][add] = _born_magnitudes(
            rng, n, np.broadcast_to(SIGN[:, None], (N_SRC, N_TGT))[add])
        h["delay"][add] = rng.integers(
            np.broadcast_to(DLY_LO[None, :], (N_SRC, N_TGT))[add],
            np.broadcast_to(DLY_HI[None, :] + 1, (N_SRC, N_TGT))[add])

    prune = h["mask"] & (rng.random((N_SRC, N_TGT)) < p_prune)
    h["mask"][prune] = False

    m = h["mask"] & (~PIN_DELAY) & (rng.random((N_SRC, N_TGT)) < p_delay)
    if m.any():
        step = rng.choice([-1, 1], int(m.sum()))
        if DELAY_LEVELS is None:
            lo = np.broadcast_to(DLY_LO[None, :], (N_SRC, N_TGT))[m]
            hi = np.broadcast_to(DLY_HI[None, :], (N_SRC, N_TGT))[m]
            h["delay"][m] = np.clip(h["delay"][m] + step, lo, hi)
        else:
            # +-1 LEVEL, which on the odd grid is +-2 ticks, clamped to the grid's ends
            h["delay"][m] = _hop(h["delay"][m], DELAY_LEVELS, step).astype(np.int64)

    m = h["mask"] & (rng.random((N_SRC, N_TGT)) < p_weight)
    if QUANTIZED:
        # a discrete hop along the sub-grid, not a Gaussian: +-1 level 80 % of the time,
        # +-2 levels 20 %, sign fair. A Gaussian of sigma 0.045 against a 0.2 grid would round
        # back to the same level ~97 % of the time, so the weight gene would be frozen.
        n = int(m.sum())
        if n:
            steps = (np.where(rng.random(n) < 0.8, 1, 2)
                     * rng.choice([-1, 1], n)).astype(np.int64)
            npos = N_IN + N_EXC_MAX
            rows = np.nonzero(m)[0]
            is_exc = rows < npos
            w = h["weight"][m]
            out = np.empty_like(w)
            if is_exc.any():
                out[is_exc] = _hop(w[is_exc], QUANT_POS, steps[is_exc])
            if (~is_exc).any():
                out[~is_exc] = _hop(w[~is_exc], QUANT_NEG, steps[~is_exc])
            h["weight"][m] = out
    else:
        h["weight"][m] += rng.normal(0.0, sigma, int(m.sum()))
    npos = N_IN + N_EXC_MAX
    h["weight"][:npos] = np.clip(h["weight"][:npos], 0.0, 1.0)
    h["weight"][npos:] = np.clip(h["weight"][npos:], -1.0, 0.0)

    # readout calibration: per-parameter Bernoulli, the same shape as the weight step rather
    # than one coin for all twelve, so a single output dimension can be retuned on its own
    if p_affine:
        h = with_identity_affine(h)
        ma = rng.random(N_TARGET) < p_affine
        h["aff_a"] = np.clip(h["aff_a"] + ma * rng.normal(0.0, SIGMA_A, N_TARGET),
                             -AFF_A_LIM, AFF_A_LIM)
        mb = rng.random(N_TARGET) < p_affine
        h["aff_b"] = np.clip(h["aff_b"] + mb * rng.normal(0.0, SIGMA_B, N_TARGET),
                             -AFF_B_LIM, AFF_B_LIM)

    # the global inhibition dial. Gated: at p_inhcoeff 0 the gene never moves, so a run that
    # does not opt in behaves exactly as if the gene did not exist.
    if p_inhcoeff and rng.random() < p_inhcoeff:
        h["inh_coeff"] = float(np.clip(inh_coeff_of(h) + rng.normal(0.0, INH_COEFF_SIGMA),
                                       INH_COEFF_LO, INH_COEFF_HI))
    # the global synaptic gain, gated the same way
    if p_gain and rng.random() < p_gain:
        h["gain"] = float(np.clip(gain_of(h) + rng.normal(0.0, GAIN_SIGMA),
                                  GAIN_LO, GAIN_HI))
    return enforce(h)


def crossover(g1, g2, rng):
    """Uniform recombination, with HIDDEN SLOTS as coherent units.

    Each hidden slot draws one parent, and takes from it BOTH its active bit AND its whole
    outgoing row. Input rows are not slots, so they recombine per cell as before.

    WHY THE SOURCE ROW AND NOT THE TARGET COLUMN. Row-coherence and column-coherence cannot
    both hold: a hidden->hidden cell belongs to two slots at once, and if their coins disagree
    one of the two must lose. Rows win because a synapse's presence, weight and delay are all
    properties stored against its PREsynaptic neuron here, so inheriting a whole row keeps a
    neuron's entire output pattern -- what makes it a meaningful unit at all -- intact. A
    neuron's incoming edges then arrive coherently from whichever parent each of its sources
    came from, and any edge that lands on a now-inactive slot is removed by enforce().
    """
    pick = rng.random((N_SRC, N_TGT)) < 0.5           # per-cell coin, used for input rows
    slot_exc = rng.random(N_EXC_MAX) < 0.5            # per-slot coin
    slot_inh = rng.random(N_INH_MAX) < 0.5
    pick[R_EXC] = slot_exc[:, None]
    pick[R_INH] = slot_inh[:, None]

    # the readout calibration recombines per OUTPUT DIMENSION, with a_j and b_j taken
    # together: they parameterise one line, and splitting them across parents would hand the
    # child a slope from one fit and an intercept from another
    a1, b1 = affine_of(g1)
    a2, b2 = affine_of(g2)
    pick_out = rng.random(N_TARGET) < 0.5
    child = dict(mask=np.where(pick, g1["mask"], g2["mask"]),
                 delay=np.where(pick, g1["delay"], g2["delay"]),
                 weight=np.where(pick, g1["weight"], g2["weight"]),
                 act_exc=np.where(slot_exc, g1["act_exc"], g2["act_exc"]),
                 act_inh=np.where(slot_inh, g1["act_inh"], g2["act_inh"]),
                 aff_a=np.where(pick_out, a1, a2),
                 aff_b=np.where(pick_out, b1, b2),
                 # one global scalar, taken WHOLE from one parent and never averaged: the
                 # midpoint of two E/I balances is not a balance either parent ever had
                 inh_coeff=(inh_coeff_of(g1) if rng.random() < 0.5 else inh_coeff_of(g2)),
                 gain=(gain_of(g1) if rng.random() < 0.5 else gain_of(g2)))
    enforce(child)
    assert dale_ok(child)[0], "crossover produced a Dale violation"
    assert legal_ok(child)[0], "crossover produced an illegal edge"
    assert pins_ok(child)[0], "crossover produced an unpinned inh->exc delay"
    assert active_ok(child)[0], "crossover produced an edge on an inactive slot"
    return child


def bundle_coherent(child, g1, g2):
    def same(a, b):
        return ((a["mask"] == b["mask"]) & (a["delay"] == b["delay"])
                & (a["weight"] == b["weight"]))
    # only cells the child could have inherited -- enforce() legitimately clears edges whose
    # target slot went inactive in the child, and those are not incoherence
    live = active_cells(child) & LEGAL
    bad = int((~(same(child, g1) | same(child, g2)) & live).sum())
    return bad == 0, bad


# ------------------------------------------------------------------ invariants
def dale_ok(g):
    npos = N_IN + N_EXC_MAX
    bad = int((g["weight"][:npos] < 0).sum() + (g["weight"][npos:] > 0).sum())
    return bad == 0, bad


def range_ok(g):
    """Normalised magnitudes stay inside the unit half-line. -> (ok, n_violations)."""
    return int((np.abs(g["weight"]) > 1.0).sum()) == 0, int((np.abs(g["weight"]) > 1.0).sum())


def legal_ok(g):
    bad = int((g["mask"] & ~LEGAL).sum())
    return bad == 0, bad


def pins_ok(g):
    bad = int(((g["delay"] != PIN_VALUE) & PIN_DELAY & g["mask"]).sum())
    return bad == 0, bad


def delays_on_grid(g):
    """Every PRESENT delay is a legal level. -> (ok, n_violations). Vacuous when off."""
    if DELAY_LEVELS is None:
        return True, 0
    bad = int((~np.isin(g["delay"][g["mask"]], DELAY_LEVELS)).sum())
    return bad == 0, bad


def active_ok(g):
    bad = int((g["mask"] & ~active_cells(g)).sum())
    return bad == 0, bad


def delays_ok(g):
    lo = np.broadcast_to(DLY_LO[None, :], (N_SRC, N_TGT))
    hi = np.broadcast_to(DLY_HI[None, :], (N_SRC, N_TGT))
    bad = int(((g["delay"] < lo) | (g["delay"] > hi))[g["mask"]].sum())
    return bad == 0, bad


def all_ok(g):
    return dict(dale=dale_ok(g)[1], legal=legal_ok(g)[1], pins=pins_ok(g)[1],
                active=active_ok(g)[1], delays=delays_ok(g)[1], range=range_ok(g)[1],
                fanout=fanout_ok(g)[1], delay_grid=delays_on_grid(g)[1],
                in_to_inh=int(g["mask"][R_IN, C_INH].sum()),
                in_to_out=int(g["mask"][R_IN, C_OUT].sum()),
                inh_to_out=int(g["mask"][R_INH, C_OUT].sum()))


# ------------------------------------------------------------------ cost
def cap_for(n_act, floor=6, frac=0.10):
    return max(floor, int(round(frac * n_act)))


def cost_terms(g, mse, lam=0.35, mu=0.10, cap_floor=6, cap_frac=0.10):
    """fitness = MSE + lam*active_neurons + mu*sum(max(0, fanout - cap)). Lower is better.

    A SOFT cap: fan-out is never clipped, only charged for.
    """
    n = n_active(g)
    cap = cap_for(n, cap_floor, cap_frac)
    fo = g["mask"].sum(1)
    excess = int(np.maximum(0, fo - cap).sum())
    return dict(mse=float(mse), n_active=n, cap=cap, fanout_excess=excess,
                neuron_penalty=lam * n, fanout_penalty=mu * excess,
                fitness=float(mse) + lam * n + mu * excess)


def genome_stats(g):
    m = g["mask"]
    w = np.abs(g["weight"])[m]
    fo = m.sum(1)
    return dict(n_syn=int(m.sum()), n_exc=int(g["act_exc"].sum()),
                n_inh=int(g["act_inh"].sum()), n_active=n_active(g),
                e_fraction=e_fraction(g),
                max_fanout=int(fo.max()) if m.any() else 0,
                n_in_hidden=int(m[R_IN].sum()),
                n_rec=int(m[N_IN:, C_EXC].sum() + m[N_IN:, C_INH].sum()),
                n_out=int(m[:, C_OUT].sum()),
                w_mean=float(w.mean()) if w.size else 0.0)


# ------------------------------------------------------------------ seed
def seed_from_small(g_small, n_exc=8, n_inh=2, gain=GAIN):
    """Embed a [27,16] tiny_snn genome into the [67,56] layout, edge for edge.

    tiny_snn genomes carry ABSOLUTE weights, so they are divided by `gain` on the way in.
    G * w_norm therefore reproduces the original absolute weight exactly.

    Old rows 0..16 in / 17..24 exc / 25..26 inh  ->  new 0..16 / 17..24 / 57..58
    Old cols 0..7 exc / 8..9 inh / 10..15 out    ->  new 0..7   / 40..41 / 50..55
    """
    g = blank(n_exc, n_inh)
    rows_old = list(range(0, S.N_SRC))
    rows_new = ([i for i in range(N_IN)]
                + [N_IN + e for e in range(S.N_EXC)]
                + [N_IN + N_EXC_MAX + i for i in range(S.N_INH)])
    cols_old = list(range(0, S.N_TGT))
    cols_new = ([e for e in range(S.N_EXC)]
                + [N_EXC_MAX + i for i in range(S.N_INH)]
                + [N_EXC_MAX + N_INH_MAX + o for o in range(N_OUT)])
    for ro, rn in zip(rows_old, rows_new):
        for co, cn in zip(cols_old, cols_new):
            g["mask"][rn, cn] = g_small["mask"][ro, co]
            g["delay"][rn, cn] = g_small["delay"][ro, co]
            g["weight"][rn, cn] = g_small["weight"][ro, co] / gain
    return enforce(g)


def activate_all(g, rng):
    """Turn on EVERY hidden slot and sparse-wire the ones that were off.

    For the fixed-full-capacity arm: the seed's working structure is kept intact in slots
    0..7 / 0..1, and the 32 + 8 newcomers each get the same 1-3 in / 1-3 out that `grow`
    would have given them -- Dale-correct, legal, and delay-pinned where required, because
    they go through the same `_wire_new_neuron` and the same `enforce`.
    """
    was_e, was_i = g["act_exc"].copy(), g["act_inh"].copy()
    g["act_exc"][:] = True
    g["act_inh"][:] = True
    for s in np.where(~was_e)[0]:
        _wire_new_neuron(g, rng, N_IN + int(s), int(s))
    for s in np.where(~was_i)[0]:
        _wire_new_neuron(g, rng, N_IN + N_EXC_MAX + int(s), N_EXC_MAX + int(s))
    return enforce(g)


# ------------------------------------------------------------------ build
def build(genomes, device="cuda", seed=1, w_ceiling=200.0, gain=None):
    """Pack P candidates into ONE SpikingNet. Every candidate always allocates the FULL
    capacity (40 exc + 10 inh); inactive slots simply carry no synapses, so they receive
    nothing and emit nothing. That is what lets candidates of different sizes share one net
    with identical per-candidate id blocks.

    THE GAIN IS APPLIED HERE AND NOWHERE ELSE. Genome weights are normalised magnitudes; the
    engine sees `gain * w`. This is the single point where normalised space becomes current."""
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine

    P = len(genomes)
    if LIF is None:
        neuron_metas = [NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                        NeuronMeta(neuron_type=1, a=0.1, d=2.0),
                        NeuronMeta(neuron_type=2, a=0.02, d=8.0),
                        NeuronMeta(neuron_type=3, a=0.02, d=8.0)]
    else:
        # LIF: v' = -(v - v_rest)/tau + I. No quadratic, so a membrane deviation DECAYS with
        # tau instead of being pulled back inside one tick -- which is the only reason
        # inhibition can hold a cell down here. Input cells (meta 2) keep a low threshold so
        # the latency code still drives them from one injected current pulse.
        from spiky.spnet.spnet import LIFNeuronMeta
        neuron_metas = [LIFNeuronMeta(neuron_type=i, **LIF) for i in range(4)]
    counts = [P * N_EXC_MAX, P * N_INH_MAX, P * N_IN, P * N_OUT_NEURONS]
    # The engine asserts every neuron type is non-empty, so a run with no inhibitory neurons
    # cannot simply declare zero of them -- the type is dropped and the remaining ones close
    # ranks. `ids` stays length 4, with an empty array in the dropped slot, so every index
    # below is unchanged and the 40/10 path is untouched.
    keep = [i for i in range(4) if counts[i] > 0]
    spnet = SpikingNet(synapse_metas=metas_(w_ceiling),
                       neuron_metas=[neuron_metas[i] for i in keep],
                       neuron_counts=[counts[i] for i in keep],
                       initial_synapse_capacity=1 << 21,
                       summation_dtype=torch.float32)
    spnet.to_device(device)
    ids = [np.zeros(0, np.int64)] * 4
    for j, i in enumerate(keep):
        ids[i] = spnet.get_neuron_ids_by_meta(j).cpu().numpy()

    tri, wts = [], []
    for c, g in enumerate(genomes):
        E = ids[0][c * N_EXC_MAX:(c + 1) * N_EXC_MAX]
        I = ids[1][c * N_INH_MAX:(c + 1) * N_INH_MAX]
        In = ids[2][c * N_IN:(c + 1) * N_IN]
        Ou = ids[3][c * N_OUT_NEURONS:(c + 1) * N_OUT_NEURONS]
        src_ids = np.concatenate([In, E, I])
        tgt_ids = np.concatenate([E, I, Ou])
        r, col = np.nonzero(g["mask"])
        if r.size == 0:
            continue
        d = g["delay"][r, col]
        tri.append(np.stack([d - D_LO, src_ids[r], tgt_ids[col]], 1))
        gv = gain_of(g) if gain is None else gain      # None -> the genome's own gene
        w = g["weight"][r, col] * gv
        # the ONLY place inh_coeff enters: inhibitory SOURCE rows are scaled, exc untouched
        w = np.where(r >= N_IN + N_EXC_MAX, w * inh_coeff_of(g), w)
        wts.append(w.astype(np.float64))

    triples = np.concatenate(tri, 0) if tri else np.zeros((0, 3), np.int64)
    weights = np.concatenate(wts, 0) if wts else np.zeros(0, np.float64)
    key = triples[:, 1].astype(np.int64) * (1 << 22) + triples[:, 2]
    assert len(np.unique(key)) == len(key), "duplicate (src,tgt) -- genome packing is wrong"

    total = sum(counts)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=S.GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + total)))
    for _ in keep:
        ge.register_neuron_type(max_synapses=4 * N_TGT, growth_command_list=[])
    for j, i in enumerate(keep):
        t = torch.tensor(ids[i], dtype=torch.int32)
        n = t.numel()
        ge.add_neurons(neuron_type_index=j, identifiers=t,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))

    chunk = ge._grow_explicit(torch.tensor(triples, dtype=torch.int32, device=device), seed,
                              weights=torch.tensor(weights, dtype=torch.float32,
                                                   device=device))
    spnet.add_connections(chunk, seed)
    chunk.recycle()
    spnet.compile(shuffle_synapses_random_seed=None)
    return dict(spnet=spnet, ids=ids, P=P, triples=triples, weights=weights, device=device)


verify_round_trip = S.verify_round_trip


# ------------------------------------------------------------------ score
# The engine has a hard ceiling on batch x neurons x ticks: past it, process_ticks dies with
# cudaErrorInvalidValue at spnet_runtime.cu:440. Measured at pool 512 (37,376 neurons): 384
# samples works (1.38e9 element-ticks), 448 does not (1.61e9) -- so the limit is around 1.4e9,
# BELOW the 2.1e9 an int32 would imply, meaning some buffer costs more than one element per
# entry. Episodes are independent, so a larger batch is run in chunks and stitched; because
# every metric here is a MEAN OVER SAMPLES, equal-sized chunks are bit-identical to one big
# call (verified). 256 is used rather than 384 for margin -- it is the size every prior run in
# this chapter has used, and it divides the batch sizes we care about evenly.
MAX_EPISODE_BATCH = 256


def grow_run_episode(H, X, enc, current=200.0, readout_window=None):
    """-> per-TARGET aggregated first-spike value [B, P, N_TARGET], plus the raw per-neuron
    [B, P, N_OUT_NEURONS].

    Identical to tiny_snn.fast_run_episode except the output layer is N_TARGET groups of
    OUT_PER_TARGET neurons, and each group is collapsed by a FIXED aggregate. Nothing here is
    evolved and nothing mixes across groups: target d sees group d and no other.
    """
    from spiky.spnet.spnet import NeuronDataType
    sp, ids, P, dev = H["spnet"], H["ids"], H["P"], H["device"]
    B = X.shape[0]
    W = int(ro_win_() if readout_window is None else readout_window)
    NT, TI = n_ticks_(), t_in_()
    ticks = enc(X)
    cols = ids[2]
    va = np.zeros((B, TI, cols.size), np.float32)
    for b in range(B):
        for j in range(N_IN):
            va[b, ticks[b, j], j::N_IN] = current
    sp.process_ticks(n_ticks_to_process=NT, batch_size=B, n_input_ticks=TI,
                     input_values=torch.as_tensor(va, device=dev),
                     sparse_input=S._sparse_ids(cols, B, TI, dev),
                     do_train=False, do_record_voltage=False,
                     do_reset_context=True, _stdp_period=32)
    oid = torch.as_tensor(np.ascontiguousarray(ids[3], dtype=np.int32), device=dev)
    R = sp.export_neuron_data(oid, B, NeuronDataType.Spike, 0, NT - 1)
    R = R.reshape(B, P, N_OUT_NEURONS, NT)[..., NT - W:]
    wgt = torch.arange(W, 0, -1, device=R.device, dtype=R.dtype)
    raw = (W - (R.ne(0) * wgt).amax(-1)).double()            # [B, P, N_OUT_NEURONS]
    grp = raw.reshape(B, P, N_TARGET, OUT_PER_TARGET)
    agg = grp.amin(-1) if OUT_AGG == "min" else grp.mean(-1)
    return agg.cpu().numpy(), raw.cpu().numpy()


def _run_chunked_grow(H, X, enc, current, chunk=None):
    chunk = MAX_EPISODE_BATCH if chunk is None else chunk
    B = X.shape[0]
    if B <= chunk:
        return grow_run_episode(H, X, enc, current)
    n = int(np.ceil(B / chunk))
    step = int(np.ceil(B / n))
    outs = [grow_run_episode(H, X[i:i + step], enc, current) for i in range(0, B, step)]
    return (np.concatenate([o[0] for o in outs], 0),
            np.concatenate([o[1] for o in outs], 0))


def _run_chunked(H, X, enc, current, chunk=None):
    # read the module global at CALL time, not as a default argument -- a default is bound
    # when the function is defined, so `G.MAX_EPISODE_BATCH = n` from a driver would have had
    # no effect whatsoever
    chunk = MAX_EPISODE_BATCH if chunk is None else chunk
    B = X.shape[0]
    if B <= chunk:
        return S.fast_run_episode(H, X, enc, current, readout_window=READOUT_WINDOW)[0]
    n = int(np.ceil(B / chunk))
    step = int(np.ceil(B / n))                      # equal-sized chunks, so means compose
    return np.concatenate([S.fast_run_episode(H, X[i:i + step], enc, current,
                                              readout_window=READOUT_WINDOW)[0]
                           for i in range(0, B, step)], axis=0)


def score(H, X, Y, enc, genomes=None, current=200.0, readout="evolved", readout_map=None):
    """readout:
         "evolved"  y = a*raw + b from the 12 affine genes (the historical behaviour)
         "linear"   y = raw @ W + b, a full N_OUT x N_OUT least-squares map

    For "linear", pass `readout_map` to APPLY a map fitted elsewhere (that is how held-out is
    scored -- with the map fitted on TRAINING data), or leave it None to fit on the (X, Y)
    given, which is what a training round does. Fitting on the batch being selected on is
    legitimate; fitting on held-out would not be, and is never done.
    -> the dict, plus "readout_map" so a caller can carry the training fit to held-out.
    """
    """-> per-candidate metrics, with the evolved readout calibration applied.

    `genomes` supplies the per-candidate (a, b); omit it (or pass genomes without the affine
    field) and every candidate uses a=1, b=0, which is exactly tiny_snn.score. That identity
    is what lets every pre-affine checkpoint reproduce its own MSE bit for bit.

    SILENCE IS MEASURED ON THE RAW OFFSETS, not the calibrated ones: "this output never fired
    in the window" is a fact about the network, and a*32+b is not the null any more.
    """
    first, raw_neurons = _run_chunked_grow(H, X, enc, current)
    R = None
    P = first.shape[1]
    if genomes is None:
        A = np.ones((P, N_TARGET))
        Bc = np.zeros((P, N_TARGET))
    else:
        ab = [affine_of(g) for g in genomes]
        A = np.stack([x[0] for x in ab])
        Bc = np.stack([x[1] for x in ab])
    tgt = task_targets(Y, X)[:, None, :]
    if readout == "lut":
        # The first-spike time IS the prediction: it indexes the LUT directly. No scale, no
        # shift, nothing fitted -- the decode is fixed in advance and the net has to land on
        # the right bin. That is the whole point of reshaping the target this way.
        assert LUT_TABLE is not None, "readout='lut' needs set_lut_task() first"
        maps = None
        y = LUT_TABLE[np.clip(first, 0, READOUT_WINDOW).astype(np.int64)]
    elif readout == "wta":
        # WINNER-TAKE-ALL over TWO output neurons: the answer is WHICH of them fires first.
        # diagls cannot express a binary answer at all -- a scale+shift on one first-spike
        # time encodes WHEN an input fired, not which of two did -- so a run scored on diagls
        # is blind to exactly the structure the LIF veto creates. Two parameters are fitted
        # on the training batch, the same budget diagls gets: the orientation (does "output 0
        # first" mean 1 or 0) and the constant used when neither wins.
        assert OUT_PER_TARGET == 2, "readout='wta' needs --out-per-target 2"
        o0, o1 = raw_neurons[:, :, 0], raw_neurons[:, :, 1]
        undec = o0 == o1                      # both silent, or the same tick: no decision
        maps, y = [], np.empty((first.shape[0], P, 1))
        for p_ in range(P):
            win = (o0[:, p_] < o1[:, p_]).astype(np.float64)
            if readout_map is None:
                t = tgt[:, 0, 0]
                u = undec[:, p_]
                orient = 1.0 if (win[~u] == t[~u]).mean() >= 0.5 else 0.0 if (~u).any() else 1.0
                fill = float(t[u].mean()) if u.any() else float(t.mean())
            else:
                orient, fill = readout_map[p_]
            maps.append((orient, fill))
            v = win if orient else 1.0 - win
            y[:, p_, 0] = np.where(undec[:, p_], fill, v)
    elif readout == "diagls":
        # DIAGONAL least squares: 2 params per target, NO cross-target mixing. Fitted on the
        # batch handed in (a training batch), or supplied via readout_map -- the training fit
        # carried unchanged to held-out. 12 params against a 1024-sample batch, so the
        # train->held-out transfer is essentially free, and held-out never sees its own fit.
        maps = []
        y = np.empty_like(first)
        for p_ in range(P):
            if readout_map is None:
                aq, bq = analytic_affine(first[:, p_, :], tgt[:, 0, :])
            else:
                aq, bq = readout_map[p_]
            maps.append((aq, bq))
            y[:, p_, :] = aq * first[:, p_, :] + bq
    elif readout == "linear":
        # one map PER CANDIDATE -- they have different outputs, so they need different maps
        maps = []
        y = np.empty_like(first)
        for p_ in range(P):
            if readout_map is None:
                W, bb = fit_linear_readout(first[:, p_, :], tgt[:, 0, :])
            else:
                W, bb = readout_map[p_]
            maps.append((W, bb))
            y[:, p_, :] = apply_linear_readout(first[:, p_, :], W, bb)
    else:
        maps = None
        y = A[None, :, :] * first + Bc[None, :, :]          # [B, P, N_OUT]
    mse = ((y - tgt) ** 2).mean(axis=(0, 2))
    Yt = Y if TARGET_DIMS is None else Y[:, TARGET_DIMS]
    # Kendall tau ranks the six action dimensions against each other; on a single-dimension
    # task there is nothing to rank, so it is undefined rather than zero.
    tau = (np.full(P, np.nan) if N_TARGET < 2 else
           np.array([kendall_tau_b(-y[:, p, :], Yt).mean() for p in range(P)]))
    silent = (first >= ro_win_()).mean(axis=(0, 2))
    n_distinct = np.array([len(np.unique(first[:, p, :])) for p in range(P)])
    if BIT_TASK is not None:
        # a 0/1 bit has no action-space counterpart; report the classification error instead,
        # which is the number that actually means something for this target
        mse_act = np.array([float(((y[:, p, 0] > 0.5) != (tgt[:, 0, 0] > 0.5)).mean())
                            for p in range(P)])
    elif LUT_TABLE is not None:
        # y is already in action units -- the LUT decoded it. Running decode_actions on top
        # would apply the offset-space inverse a second time.
        Ya = Y[:, [LUT_DIM]]
        mse_act = np.array([((y[:, p, :] - Ya) ** 2).mean() for p in range(P)])
    else:
        mse_act = np.array([((decode_actions(y[:, p, :]) - Yt) ** 2).mean() for p in range(P)])
    return dict(fitness=-mse, mse=mse, tau=tau, silent=silent, n_distinct=n_distinct,
                mse_action=mse_act, first=first, calibrated=y, raster=R,
                raw_neurons=raw_neurons, aff_a=A, aff_b=Bc, readout_map=maps)


def fit_linear_readout(raw, tgt, lam=1.0):
    """Least-squares FULL map from all N_OUT first-spike offsets to all N_OUT targets.

    WHY THIS EXISTS. The 12 affine genes can only express a DIAGONAL map -- output d predicts
    target d and nothing else. Measured on the quantized champion, the best diagonal map
    reaches 34.73 against a chance level of 34.15: even perfectly optimised it cannot beat
    predicting the mean. The same six numbers under a full 6x6 map reach 21.73. So the
    outputs are informative but MIS-ALIGNED -- each one's timing tracks a mixture of target
    dimensions, and forcing output d onto target d throws that away.

    -> (W, b) with y = raw @ W + b.
    """
    A = np.column_stack([raw, np.ones(len(raw))])
    n = A.shape[1]
    M = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ tgt)
    return M[:-1], M[-1]


def apply_linear_readout(raw, W, b):
    return raw @ W + b


def analytic_affine(raw, tgt):
    """Closed-form least-squares (a, b) per output dimension: t ~ a*raw + b.

    Fit on whatever data is passed in. Fitting on the HELD-OUT set reproduces the reported
    `affine_ceiling` exactly and is therefore an optimistic bound; fitting on a TRAINING batch
    and scoring on held-out is the honest number an evolved calibration could actually reach.
    """
    a = np.zeros(raw.shape[1])
    b = np.zeros(raw.shape[1])
    for d in range(raw.shape[1]):
        p, t = raw[:, d], tgt[:, d]
        v = p.var()
        a[d] = float(np.cov(p, t, bias=True)[0, 1] / v) if v > 1e-12 else 0.0
        b[d] = float(t.mean() - a[d] * p.mean())
    return a, b
