"""exp012 — the three-stage spiking LUT rebuilt for the QUANTISED PPO policy.

A FORK of tiny_lut_full_pipeline.py (which is left byte-identical). Two changes:

  1. INPUT ENCODER -> the shipped 128-bucket SHARED Gaussian-companded map.
     tick = (T_IN-1) - searchsorted(in_quant_edges, x_norm)
     Larger value -> larger bucket -> EARLIER tick, which is the convention the
     comparators already use. Only `in_quant_edges` crosses into the SNN: the network
     consumes tick ORDERING only, never the dequantised value, so `in_quant_dequant`
     stays a software-side artefact. Ordering is preserved exactly, including ties, so
     the address bits are the SAME function as the software LUT's `d > 0` -- not an
     approximation of it. Stage 1/2 timing, t_in and GATE_TICK are untouched.

     The tie detectors STAY. The Gaussian map ties 0.950% of address slots and 77.12% of
     samples carry at least one; on a tick-tie both rails fire (`loser_false_fire` ==
     `tie_frac` in the recorded ablation), so without the veto both memories latch and the
     table multi-selects. Removing them is separate work.

  2. STAGE 3 -> AMPLITUDE ENCODING instead of delays. This is the substantive change.

     Today each selected cell reaches the output neuron after a delay encoding its weight,
     and the anti-leak membrane turns the spread of arrival times into a logsumexp. The
     delay span (75.8 ticks here, 236 at 22 levels) is the whole Stage-3 latency and the
     thing that scrapes the engine's 255-tick cap.

     Because the output neuron is exactly linear (cf_2 = 0, a=b=c=d=0), the same readout is
     available with ZERO delay spread. With arrivals a_t = A - s*w_t, s = tau_eff/tau, and
     one shared amplitude alpha, the drive is

         alpha * e^{t/tau_eff} * SUM_t e^{-a_t/tau_eff}
           = alpha * e^{-A/tau_eff} * e^{t/tau_eff} * SUM_t e^{w_t/tau}        [s/tau_eff = 1/tau]

     which is exactly what you get by putting every spike on ONE tick with per-synapse
     weight beta_o * e^{w_t/tau}, beta_o = alpha_o * e^{-A/tau_eff}. Same logsumexp, same
     decode SLOPE (-32*tau/tau_eff action units per tick); only the offset moves.

     It is also strictly MORE faithful: the delay path rounds every delay with rint(), so it
     evaluates a quantised logsumexp. Measured on the shipped weights, that rounding is worth
     up to 0.25 tick, which the amplitude form recovers.

     And it stops depending on the one kernel limitation we cannot change: the delay path
     needs the membrane to integrate arrivals ACROSS ticks, whereas here everything lands in
     a single tick where `I` is summed before being applied, so `I` being zeroed next tick is
     irrelevant. No synaptic decay constant is required.

     Consequence: Stage-3 delay span = 1 tick, dmax ~3 (Stage 1/2 only), and raising
     TAU_M_OUT for finer output levels now costs only a longer CROSSING WINDOW (~21 ticks for
     22 levels) instead of ~236 ticks of delay.

The amplitude vernier (K copies per dim) is deliberately NOT included -- plain amplitude
encoding first, then decide the vernier by measurement.
"""
import argparse
import json
import os

import numpy as np
import torch

from tiny_lut_order_full import pair_list

T_IN = 128
DE, DI, W_EXC, W_INH = 3, 2, 1.5, -10.0
TAU_M_RAIL, TAU_MEM, W_MEM, W_GATE = 20.0, 1200.0, 0.6, 0.6
W_TIE, TH_TIE = 0.6, 1.0
W_AND = 0.18
# 140, matching the deployed actor: it only has to clear the 128-tick input window
# plus rail settling. The dev pipeline used 200, which cost 60 ticks for nothing.
GATE_TICK, EMIT = 140, 143
D_OUT = 1                      # the ONLY Stage-3 delay now

# Completion detector (Change A). Each of the 17 inputs fires EXACTLY ONCE, so a perfectly
# leak-free neuron fed by all of them crosses threshold precisely when the LAST one arrives.
# W_DET must satisfy 16*W < 1 <= 17*W, i.e. W in [1/17, 1/16) = [0.0588, 0.0625).
#
# It has to be a genuinely leak-free NeuronMeta (cf_1 = 0), NOT the tau=1200 memory type:
# with tau=1200 the worst case (16 inputs at tick 0, the 17th at tick 127) decays the early
# ones by e^{-127/1200} = 0.898, giving 0.062*(16*0.898 + 1) = 0.953 < 1.0 -- it would fail
# to fire on exactly the spread inputs it exists to detect.
W_DET = 0.06
# The detector fires at t_last+1, but the LAST rail only fires at t_last+DE = t_last+3, its
# tie detector at +4, and that tie veto reaches memory at +5. So the gate pulse must arrive
# AFTER +5, or memory latches before the last comparator has decided. Measured directly:
# with delay 1 the gate regressed to 99.9196% bit parity and 78 multi-selects. Hence 6.
D_GATE = 6
# GT-SKEW (replaces the 136 tie-detector neurons).
#
# Measured on the real kernel (tiny_tie_trace.py): the cross-inhibition lands exactly ONE
# TICK AFTER the excitation it must cancel. So on an exact tie both rails see pure +1.5187
# and BOTH fire (the veto is late); 1 tick apart, the loser sees +1.5-10 = -8.5 in a single
# tick and never rises. That is why ties needed a separate detector.
#
# Adding one tick to the GT rail's excitation aligns the tie case with the veto without
# touching the 1-apart case:
#   tie        -> GT excitation coincides with the LT inhibition -> -8.5, silent -> bit 0
#   a earlier  -> GT fires, LT inhibition not until +2           -> bit 1
#   a later    -> GT meets a runaway-negative membrane           -> bit 0
# which is exactly the software's strict `d > 0`. It is structural, not tuned: the gap
# between "fires" (+1.519) and "silent" (-8.606) is 10.1 against a threshold of 1.0.
#
# All 136 pairs have anchor_a as the LOWER index (measured), so the GT rail is always r0 --
# but the skew is applied through the pair's own orientation below, not by index, so it does
# not depend on that accident.
DE_GT = DE + 1


def calib(tau_m, n_euler=2, dt=0.5):
    return 1.0 / np.log((1.0 + dt / tau_m) ** n_euler)


# ---------------------------------------------------------------------------------------
# input encoding
# ---------------------------------------------------------------------------------------
def encode_gauss(x_norm, edges, t_in=T_IN):
    """Shared Gaussian-companded latency code. Larger value -> earlier tick."""
    g = np.searchsorted(edges, np.asarray(x_norm, np.float64).ravel(), side="left")
    g = g.reshape(np.shape(x_norm))
    return np.clip((t_in - 1) - g, 0, t_in - 1).astype(np.int64)


# ---------------------------------------------------------------------------------------
# Stage-3 amplitude calibration
# ---------------------------------------------------------------------------------------
def stage3_amplitudes(W, tau, tau_eff, dims, cross_at=3.0, clip=1.0):
    """Per-dim beta_o and the decode affine.

    Crossing (threshold 1, arrival at tick A0):  n = tau_eff * log(1 / (beta * S)),
    with S = SUM_t e^{w_t/tau}. The teacher's readout is out = 32*tau*log(S/32), so
    S(out) = 32 * exp(out / (32*tau)).

    beta_o is set so the TOP of the actuator band (out = +clip) crosses `cross_at` ticks
    after arrival -- any earlier and the top of the range saturates at the arrival tick
    itself and is unrecoverable.
    """
    beta, aff, win = {}, {}, {}
    for o in dims:
        s_hi = 32.0 * np.exp(+clip / (32.0 * tau))          # drive at out = +clip
        s_lo = 32.0 * np.exp(-clip / (32.0 * tau))          # drive at out = -clip
        beta[o] = float(np.exp(-cross_at / tau_eff) / s_hi)
        n_hi = tau_eff * np.log(1.0 / (beta[o] * s_hi))     # == cross_at
        n_lo = tau_eff * np.log(1.0 / (beta[o] * s_lo))
        # n = const - (tau_eff/(32 tau)) * out  =>  out = (const - n) * 32 tau / tau_eff
        slope = -32.0 * tau / tau_eff
        offset = +clip - slope * n_hi
        beta[o], aff[o], win[o] = beta[o], (slope, offset), (n_hi, n_lo)
    return beta, aff, win


def build(Z, dims, tie_break, tau_m_out=10.0, device="cuda", margin=6,
          cross_at=3.0, gt_skew=False):
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    pairs, slot = pair_list(Z)
    P = len(pairs)
    W = Z["weights"].astype(np.float64)
    tau = float(Z["tau_actor"])
    tau_eff = calib(tau_m_out)
    clip = float(Z["out_quant_clip"]) if "out_quant_clip" in Z.files else 1.0
    beta, aff, win = stage3_amplitudes(W, tau, tau_eff, dims, cross_at, clip)

    dmax = max(DE, DE_GT, DI, D_OUT, D_GATE)
    assert dmax <= 255, f"delay {dmax} exceeds the engine's 255-tick synapse limit"
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=64, _backward_group_size=64)
              for d in range(1, dmax + 1)]
    NT = len(dims)
    metas = [LIFNeuronMeta(neuron_type=0, tau=TAU_M_RAIL, threshold=1.0),
             NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / TAU_M_RAIL, cf_0=0.0, a=0.0,
                        b=0.0, c=0.0, d=0.0, spike_threshold=1.0),
             LIFNeuronMeta(neuron_type=2, tau=TAU_M_RAIL, threshold=1.0),
             LIFNeuronMeta(neuron_type=3, tau=TAU_MEM, threshold=1.0),
             *([] if not tie_break else
               [LIFNeuronMeta(neuron_type=4, tau=TAU_M_RAIL, threshold=TH_TIE)]),
             # Change B: leak-free (g=1) so the 6-way AND no longer depends on the
             # arrival phase -- a wrong cell is bounded by 5*0.18 = 0.90 for ALL time.
             NeuronMeta(neuron_type=5, cf_2=0.0, cf_1=0.0, cf_0=0.0, a=0.0, b=0.0,
                        c=0.0, d=0.0, spike_threshold=1.0),
             NeuronMeta(neuron_type=6, cf_2=0.0, cf_1=+1.0 / tau_m_out, cf_0=0.0, a=0.0,
                        b=0.0, c=0.0, d=0.0, spike_threshold=1.0),
             NeuronMeta(neuron_type=7, cf_2=0.0, cf_1=0.0, cf_0=0.0, a=0.0, b=0.0,
                        c=0.0, d=0.0, spike_threshold=1.0)]        # completion detector
    counts = ([18, 2 * P, 2 * P, 2 * P] + ([P] if tie_break else [])
              + [2048, NT, 1])
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=counts,
                     initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
    net.to_device(device)
    NTY = 8 if tie_break else 7
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(NTY)]
    if not tie_break:                      # keep index names stable downstream
        ids = ids[:4] + [np.zeros(0, np.int64)] + ids[4:]
    inp = ids[0][:17]
    det = ids[7][0]                     # Change A: the internal gate
    E = [(1, inp[j], det, W_DET) for j in range(17)]
    for p, (a, b) in enumerate(pairs):
        r0, r1 = ids[1][2 * p], ids[1][2 * p + 1]
        i0, i1 = ids[2][2 * p], ids[2][2 * p + 1]
        m0, m1 = ids[3][2 * p], ids[3][2 * p + 1]
        # r0 is the GT rail (driven by the pair's anchor_a); it carries the +1 skew.
        de_gt, de_lt = (DE_GT, DE) if gt_skew else (DE, DE)
        E += [(de_gt, inp[a], r0, W_EXC), (de_lt, inp[b], r1, W_EXC),
              (DI, inp[a], i0, W_EXC), (DI, inp[b], i1, W_EXC),
              (1, i0, r1, W_INH), (1, i1, r0, W_INH),
              (1, r0, m0, W_MEM), (1, r1, m1, W_MEM),
              (D_GATE, det, m0, W_GATE), (D_GATE, det, m1, W_GATE)]
        if tie_break:
            td = ids[4][p]
            E += [(1, r0, td, W_TIE), (1, r1, td, W_TIE), (1, td, m0, -W_MEM)]
    for t in range(32):
        for k in range(64):
            cell = ids[5][t * 64 + k]
            for j in range(6):
                bit = (k >> (5 - j)) & 1
                p, a_first = slot[t * 6 + j]
                r1i = 2 * p + (0 if a_first else 1)
                E.append((1, ids[3][r1i if bit else (2 * p + (1 if a_first else 0))],
                          cell, W_AND))
            # ---- AMPLITUDE-ENCODED STAGE 3: one delay, weight carries the value ----
            for oi, o in enumerate(dims):
                E.append((D_OUT, cell, ids[6][oi],
                          beta[o] * float(np.exp(W[t, k, o] / tau))))
    tri = np.array([[d - 1, s, tg] for d, s, tg, _ in E], np.int64)
    wts = np.array([w for *_, w in E], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=64,
                             max_groups_in_buffer=max(1 << 19, 4 * len(tri)))
    for i in range(NTY):
        ge.register_neuron_type(max_synapses=1 << 15, growth_command_list=[])
    for i, gi in enumerate([j for j in range(8) if len(ids[j])]):
        tt = torch.tensor(ids[gi], dtype=torch.int32)
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(tt.numel()).float(),
                                                torch.zeros(tt.numel()),
                                                torch.full((tt.numel(),), float(i))], 1))
    chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32, device=device), 1,
                              weights=torch.tensor(wts, dtype=torch.float32, device=device))
    net.add_connections(chunk, 1)
    chunk.recycle()
    net.compile(shuffle_synapses_random_seed=None)
    # the episode now ends just after the BOTTOM of the band crosses; anything later is
    # out-of-band low and clips to -1 anyway, so silence IS the correct answer there.
    # data-dependent now: the detector fires at t_last+1, memory at +2, cell at +3, the
    # output receives at +4 and crosses within the window. Size the buffer for the worst
    # case (t_last = T_IN-1); the EFFECTIVE episode per sample is measured in main().
    n_ticks = int((T_IN - 1) + D_GATE + 4 + max(w[1] for w in win.values()) + margin)
    return net, ids, len(tri), n_ticks, sum(counts), aff, win, beta, dmax


def run(net, ids, ticks, n_ticks, device="cuda"):
    from spiky.spnet.spnet import NeuronDataType
    B = ticks.shape[0]
    va = np.zeros((B, n_ticks, 18), np.float32)
    for j in range(17):
        va[np.arange(B), ticks[:, j], j] = 1e6
    # Change A: NO external gate stimulus -- the completion detector is the gate.
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, n_ticks, -1).contiguous()
    net.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    out = {}
    for k in (3, 5, 6):
        oid = torch.as_tensor(np.ascontiguousarray(ids[k], dtype=np.int32), device=device)
        R = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
        R = R.reshape(B, len(ids[k]), n_ticks).ne(0)
        if k == 6:
            w = torch.arange(n_ticks, 0, -1, device=R.device, dtype=torch.float32)
            # n_ticks means SILENCE -> out-of-band low -> clips to -1
            out[k] = (n_ticks - (R.float() * w).amax(-1)).cpu().numpy().astype(np.int64)
        else:
            out[k] = R.sum(-1).cpu().numpy()
    return out


def main():
    # Resolve the defaults relative to this file, the way the sibling scripts do, so a clone
    # or a worktree anywhere finds the committed data/ and the exp19 teacher without flags.
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        here, "..", "exp19_lut-lse-expmlpcrit-t32", "deploy", "quantised",
        "walker2d_fastlut_lse_exp19_quantised.npz"))
    ap.add_argument("--data", default=os.path.join(
        here, "data", "distill_exp19_100k.npz"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--dims", default="0,1,2,3,4,5")
    ap.add_argument("--tau-m-out", type=float, default=10.0)
    ap.add_argument("--cross-at", type=float, default=3.0)
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--no-tie-break", action="store_true")
    ap.add_argument("--gt-skew", action="store_true",
                    help="+1 tick on the GT rail; replaces the tie detectors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dims = [int(v) for v in a.dims.split(",")]

    Z = np.load(a.npz)
    D = np.load(a.data)
    tau = float(Z["tau_actor"])
    edges = Z["in_quant_edges"].astype(np.float64)
    dequant = Z["in_quant_dequant"].astype(np.float64)
    LV, CLIP = int(Z["out_quant_levels"]), float(Z["out_quant_clip"])
    STEP = 2 * CLIP / (LV - 1)

    # observations -> normalised -> ticks (held-out tail, as the original does)
    raw = D["x"].astype(np.float64)[-a.n:]
    raw[:, 8:] = np.clip(raw[:, 8:], -10, 10)
    xn = (raw - Z["obs_mean"]) / np.sqrt(Z["obs_var"] + 1e-8)
    ticks = encode_gauss(xn, edges)

    # ---- the SOFTWARE reference: what the shipped quantised actor emits ----------------
    xq = dequant[np.clip((T_IN - 1) - ticks, 0, T_IN - 1)]
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    bits_sw = (xq[:, A_] - xq[:, B_]) > 0
    idx_sw = (bits_sw.astype(np.int64) * (1 << np.arange(A_.shape[1] - 1, -1, -1))).sum(-1)
    sel = Z["weights"].astype(np.float64)[np.arange(32)[None, :], idx_sw]
    lse = np.log(np.exp(sel / tau).mean(1))
    mu_sw = 32 * tau * lse
    q_sw = np.clip(np.round((np.clip(mu_sw, -CLIP, CLIP) + CLIP) / STEP) * STEP - CLIP,
                   -CLIP, CLIP)

    net, ids, nsyn, n_ticks, nneur, aff, win, beta, dmax = build(
        Z, dims, not a.no_tie_break, a.tau_m_out, a.device, a.margin, a.cross_at,
        a.gt_skew)
    print(f"neurons {nneur}  synapses {nsyn}  n_ticks {n_ticks}  dmax {dmax}  "
          f"tau_m_out {a.tau_m_out}  levels_representable "
          f"{int(2*CLIP/(32*tau/calib(a.tau_m_out)))+1}")
    print(f"crossing window per dim (ticks after arrival): "
          f"{ {o: (round(w[0],2), round(w[1],2)) for o, w in win.items()} }")

    M, C, T = [], [], []
    for s in range(0, len(ticks), a.chunk):
        o = run(net, ids, ticks[s:s + a.chunk], n_ticks, a.device)
        M.append(o[3]); C.append(o[5]); T.append(o[6])
    M, C, T = np.concatenate(M), np.concatenate(C), np.concatenate(T)

    # ---- Stage 1: address-bit parity ---------------------------------------------------
    pairs, slot = pair_list(Z)
    bits_snn = np.zeros_like(bits_sw)
    for t in range(32):
        for j in range(6):
            p, a_first = slot[t * 6 + j]
            gt = 2 * p + (0 if a_first else 1)
            bits_snn[:, t, j] = M[:, gt] > 0
    s1 = float((bits_snn == bits_sw).mean())
    # CHECK B: split parity by whether the slot was a TICK-TIE -- a wrong skew direction
    # fails only here, and would be invisible in the aggregate.
    tie_slot = ticks[:, A_] == ticks[:, B_]
    s1_tie = float((bits_snn == bits_sw)[tie_slot].mean()) if tie_slot.any() else float("nan")
    s1_non = float((bits_snn == bits_sw)[~tie_slot].mean())
    # 1-tick-apart genuine comparisons, the tightest non-tie
    near = np.abs(ticks[:, A_].astype(int) - ticks[:, B_].astype(int)) == 1
    s1_near = float((bits_snn == bits_sw)[near].mean()) if near.any() else float("nan")
    print(f"CHECK B  tied slots      : {s1_tie*100:.4f}%  (n={int(tie_slot.sum()):,}, "
          f"{tie_slot.mean()*100:.2f}% of slots; samples with >=1 tie "
          f"{tie_slot.any((1,2)).mean()*100:.2f}%)")
    print(f"CHECK B  non-tied slots  : {s1_non*100:.4f}%  (n={int((~tie_slot).sum()):,})")
    print(f"CHECK B  1-tick-apart    : {s1_near*100:.4f}%  (n={int(near.sum()):,})")
    print(f"CHECK D  census          : {nneur} neurons, {nsyn} synapses, dmax {dmax}")

    # ---- Stage 2: one-hot ---------------------------------------------------------------
    per_table = C.reshape(len(C), 32, 64).sum(-1)
    none_, multi = int((per_table == 0).sum()), int((per_table > 1).sum())

    # ---- Stage 3: EXACT-MATCH on the 22-level grid --------------------------------------
    # The absolute arrival tick depends on the exact Stage-1/2 spike timing, which is an
    # engine detail rather than something to assume. So the SLOPE is taken from theory
    # (-32*tau/tau_eff, the whole point of the derivation) and only the OFFSET is fitted,
    # on the first half and applied to the second -- the same discipline the original
    # pipeline uses. The fitted-vs-theory slope is reported as an independent check.
    t_last = ticks.max(1)
    half = len(T) // 2
    res = {}
    for oi, o in enumerate(dims):
        sl, _ = aff[o]
        # Self-timing makes the whole downstream chain data-dependent, so the crossing
        # tick is no longer measured against a fixed origin. Stage 3 is an ABSOLUTE-time
        # code, so it needs a reference: the detector's own firing tick, which is
        # t_last + 1 and is known from the input encoding at zero cost.
        # HALF-TICK DEBIAS. T is the ceil of the continuous crossing time and the decode
        # slope is NEGATIVE, so rounding the tick UP rounds the action DOWN -- measured on
        # 5,120 on-policy states as 100% one-sided -1 errors at ~21%, mean -0.207 levels.
        # The unbiased estimate of the continuous crossing is t ~ T - 0.5.
        n_after = (T[:, oi] - t_last - 0.5).astype(np.float64)
        live = T[:, oi] < n_ticks
        fit = live.copy(); fit[half:] = False
        off = float(np.median(mu_sw[fit, o] - sl * n_after[fit])) if fit.any() else 0.0
        sl_fit = (np.polyfit(n_after[fit], mu_sw[fit, o], 1)[0] if fit.sum() > 2 else np.nan)
        # PHASE alignment. |slope| = 32*tau/tau_eff and the target level step 2*clip/(L-1)
        # are the same number by construction once tau_m_out is chosen for L levels, so the
        # spiking tick grid and the software level grid are the same lattice up to a shift.
        # Getting that shift wrong costs a whole level on ~half the samples for no reason.
        # Search it on the FIRST half only and apply to the second.
        # NOTE: no exact-match phase search here, deliberately. Maximising exact-match
        # re-selects the biased-low offset (verified: the search returned a 0.0 shift on the
        # on-policy set while the mean signed residual sat at -0.207 levels). The mean fit
        # above centres the residual instead, which is the objective that matters for return.
        aff[o] = (sl, off)
        mu = sl * n_after + off
        mu = np.where(~live, -CLIP, mu)                        # silence -> -1
        q = np.clip(np.round((np.clip(mu, -CLIP, CLIP) + CLIP) / STEP) * STEP - CLIP,
                    -CLIP, CLIP)
        res[o] = dict(exact=float((np.abs(q - q_sw[:, o]) < 1e-9).mean()),
                      within1=float((np.abs(q - q_sw[:, o]) < STEP * 1.5).mean()),
                      maxabs=float(np.abs(q - q_sw[:, o]).max()),
                      silent=float((T[:, oi] >= n_ticks).mean()))
    eff = np.where(T.max(1) < n_ticks, T.max(1) + 1, n_ticks)
    print(f"\nEPISODE (data-dependent): min {eff.min()}  mean {eff.mean():.1f}  "
          f"max {eff.max()}  (fixed buffer {n_ticks}); last-input tick "
          f"min {t_last.min()} mean {t_last.mean():.1f} max {t_last.max()}")
    print(f"STAGE 1 bit parity : {s1*100:.4f}%   ({int((bits_snn!=bits_sw).sum())} bad of "
          f"{bits_sw.size})")
    print(f"STAGE 2 one-hot    : {none_} none, {multi} multi of {per_table.size}")
    print(f"STAGE 3 exact match on the {LV}-level grid:")
    for o in dims:
        r = res[o]
        print(f"  dim {o}: exact {r['exact']*100:7.3f}%  within-1-level "
              f"{r['within1']*100:7.3f}%  max|err| {r['maxabs']:.4f}  "
              f"silent {r['silent']*100:.2f}%")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(npz=a.npz, n=int(a.n), tau_m_out=a.tau_m_out, n_ticks=int(n_ticks),
                       episode_min=int(eff.min()), episode_mean=float(eff.mean()),
                       episode_max=int(eff.max()),
                       dmax=int(dmax), neurons=int(nneur), synapses=int(nsyn),
                       levels=LV, stage1_bit_parity=s1, stage2_none=none_,
                       stage2_multi=multi, stage3={str(k): v for k, v in res.items()},
                       affine={str(k): list(v) for k, v in aff.items()},
                       beta={str(k): v for k, v in beta.items()}),
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
