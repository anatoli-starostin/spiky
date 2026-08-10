"""Drive a random chapter-substrate SPNet with real dataset input under STDP for a long run,
then render one spike raster of a 1-second window across all four neuron bands.

THE QUESTION: after long STDP exposure to real input, does the net develop internal structure
-- polychronous groups, population rhythms, anything beyond a stimulus-locked echo?

SUBSTRATE is the chapter's, unchanged: steady_state.seed_genome + build_pool at stdp_lr > 0,
so excitatory synapses are plastic and inhibitory ones are frozen at RES_W_INH = -5
(stage2_metas). 800 excitatory + 200 inhibitory + 17 input + 6 output.

DRIVE is the chapter's, unchanged: LatencyEncoder over the distillation dataset, each state's
17 features encoded as one first-spike time in [0, T_IN) and injected as current on the input
neurons. States are streamed back to back, EPISODE ticks apart, and the context is reset ONCE
at the start rather than per episode -- the run is meant to be 3,600 continuous seconds, and a
reset every 96 ticks would erase exactly the slow structure we are looking for.

The final window is captured with do_train=False: the picture should show what the trained net
does, not what it is in the middle of becoming.

    python raster_long_stdp.py --ticks 3600000 --out /tmp/raster.png
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import steady_state as S                                  # noqa: E402
from data import load                                     # noqa: E402
from harness import LatencyEncoder, N_IN, N_OUT, N_TICKS, T_IN   # noqa: E402

EPISODE = N_TICKS          # ticks per dataset state, as in run_episode
BANDS = ("input", "hidden exc", "hidden inh", "output")
# chapter analytics palette (src/analytics/*), plus two more Tableau steps
COLORS = {"input": "#79706E", "hidden exc": "#4E79A7",
          "hidden inh": "#E15759", "output": "#59A14F"}


def make_stream(enc, X, ids_in, n_ticks, rng, current, device):
    """[1, n_ticks, N_IN] input values + matching neuron ids: one dataset state per EPISODE."""
    n_states = -(-n_ticks // EPISODE)                      # ceil: never leave a dead tail
    pick = rng.integers(0, X.shape[0], n_states)
    first = enc(X[pick])                                   # [n_states, N_IN] in [0, T_IN)
    va = np.zeros((1, n_ticks, N_IN), np.float32)
    t = (np.arange(n_states)[:, None] * EPISODE + first).ravel()
    j = np.tile(np.arange(N_IN), n_states)
    keep = t < n_ticks
    va[0, t[keep], j[keep]] = current
    sp_ids = np.broadcast_to(ids_in.reshape(1, 1, N_IN), (1, n_ticks, N_IN))
    return (torch.tensor(va, device=device),
            torch.tensor(sp_ids.copy().astype(np.int32), device=device))


def exc_weights(sp, ids, device):
    """Weights of the PLASTIC (excitatory-bank) synapses only, keyed for a before/after diff."""
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device=device)
    n = int(sp.count_synapses(all_ids, True))
    b = [torch.zeros(n, dtype=t, device=device) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    s, m, w, d, t = (x.cpu().numpy() for x in b)
    k = np.lexsort((m, t, s))
    s, m, w = s[k], m[k], w[k]
    return w, (m < S.N_DELAY_METAS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3_600_000, help="STDP training ticks")
    ap.add_argument("--chunk", type=int, default=9600, help="ticks per process_ticks call")
    ap.add_argument("--window", type=int, default=1000, help="ticks in the captured raster")
    ap.add_argument("--stdp-lr", type=float, default=0.01)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raster_long_stdp.png"))
    ap.add_argument("--report-every", type=int, default=25, help="chunks between log lines")
    a = ap.parse_args()

    from spiky.spnet.spnet import NeuronDataType

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # ---- substrate: the chapter's own genome and builder, one net
    g = S.seed_genome(np.random.default_rng(a.seed), a.w_max)
    h = S.build_pool([g], a.device, seed=1, stdp_lr=a.stdp_lr, w_max=a.w_max)
    sp, ids = h["spnet"], h["ids"]
    counts = dict(zip(BANDS, (N_IN, S.N_EXC, S.N_INH, N_OUT)))
    print(f"net: {counts}  total {sum(counts.values())} neurons, "
          f"{h['n_syn']:,} synapses, {2 * S.N_DELAY_METAS} metas "
          f"(exc plastic lr={a.stdp_lr}, inh frozen at {S.RES_W_INH})", flush=True)

    # ---- drive: the chapter's encoder over the real dataset
    _X, _Y, Xpool, _Yp, Xval, _Yv = load(64, a.seed, 2000)
    enc = LatencyEncoder(Xpool)
    print(f"dataset: pool {Xpool.shape}, held-out {Xval.shape}; "
          f"LatencyEncoder over the pool, {EPISODE} ticks per state", flush=True)

    w0, exc_mask = exc_weights(sp, ids, a.device)
    print(f"before training: {int(exc_mask.sum()):,} plastic synapses, "
          f"mean w {w0[exc_mask].mean():.4f}, std {w0[exc_mask].std():.4f}", flush=True)

    order = np.concatenate([ids[2], ids[0], ids[1], ids[3]])       # in, exc, inh, out
    order_t = torch.tensor(order, dtype=torch.int32, device=a.device)
    edges = np.cumsum([0] + [counts[b] for b in BANDS])

    def capture(reset, cap_rng):
        """One window on held-out states with STDP OFF -> [n_neurons, window] bool + stats."""
        va, spi = make_stream(enc, Xval, ids[2], a.window, cap_rng, a.current, a.device)
        sp.process_ticks(n_ticks_to_process=a.window, batch_size=1, n_input_ticks=a.window,
                         input_values=va, sparse_input=spi, do_train=False,
                         do_record_voltage=False, do_reset_context=reset, _stdp_period=32)
        R = (sp.export_neuron_data(order_t, 1, NeuronDataType.Spike,
                                   0, a.window - 1)[0].cpu().numpy() > 0)
        st = {}
        for i, b in enumerate(BANDS):
            sub = R[edges[i]:edges[i + 1]]
            st[b] = dict(n=sub.shape[0], spikes=int(sub.sum()),
                         hz=float(sub.sum() / sub.shape[0] / (a.window / 1000.0)),
                         silent=int((~sub.any(1)).sum()))
        return R, st

    # UNTRAINED CONTROL, same states, same seed: without it "the trained net has rhythms" is
    # unfalsifiable -- a random reservoir driven by a periodic stimulus already looks periodic.
    R_pre, stats_pre = capture(True, np.random.default_rng(a.seed + 777))
    print(f"untrained control window: {int(R_pre.sum()):,} spikes  "
          + "  ".join(f"{b} {stats_pre[b]['hz']:.1f} Hz" for b in BANDS), flush=True)

    # ---- long STDP run
    n_chunks = max(1, a.ticks // a.chunk)
    print(f"training {n_chunks * a.chunk:,} ticks in {n_chunks} chunks of {a.chunk:,} "
          f"({a.chunk // EPISODE} states each), STDP on, context reset once at the start",
          flush=True)
    t0 = time.time()
    for c in range(n_chunks):
        va, spi = make_stream(enc, Xpool, ids[2], a.chunk, rng, a.current, a.device)
        sp.process_ticks(n_ticks_to_process=a.chunk, batch_size=1, n_input_ticks=a.chunk,
                         input_values=va, sparse_input=spi, do_train=True,
                         do_record_voltage=False, do_reset_context=(c == 0), _stdp_period=32)
        if (c + 1) % a.report_every == 0 or c == n_chunks - 1:
            torch.cuda.synchronize()
            el = time.time() - t0
            done = (c + 1) * a.chunk
            print(f"  {done:,}/{n_chunks * a.chunk:,} ticks  {el / 60:.1f} min  "
                  f"{done / el:,.0f} ticks/s  eta {(n_chunks * a.chunk - done) / (done / el) / 60:.1f} min",
                  flush=True)
    torch.cuda.synchronize()
    train_s = time.time() - t0
    print(f"training done in {train_s / 60:.1f} min ({a.ticks / train_s:,.0f} ticks/s)",
          flush=True)

    w1, _ = exc_weights(sp, ids, a.device)
    dw = w1[exc_mask] - w0[exc_mask]
    moved = int((np.abs(dw) > 1e-6).sum())
    print(f"after training: mean w {w1[exc_mask].mean():.4f} (was {w0[exc_mask].mean():.4f}), "
          f"moved {moved:,}/{int(exc_mask.sum()):,} ({100 * moved / exc_mask.sum():.1f} %), "
          f"max|dw| {np.abs(dw).max():.4f}, mean|dw| {np.abs(dw).mean():.4f}", flush=True)
    inh = ~exc_mask
    print(f"inhibitory control: {int(inh.sum()):,} frozen synapses, "
          f"all still {S.RES_W_INH}: {bool(np.allclose(w1[inh], S.RES_W_INH))}", flush=True)

    # ---- capture window, still on real input, STDP off, SAME states as the control
    R, stats = capture(False, np.random.default_rng(a.seed + 777))
    np.savez_compressed(os.path.splitext(a.out)[0] + "_raster.npz", raster=R,
                        raster_untrained=R_pre,
                        counts=np.array([counts[b] for b in BANDS]))

    print(f"\nwindow: {a.window} ticks = {a.window / 1000:.1f} s, "
          f"{int(R.sum()):,} spikes total (untrained control {int(R_pre.sum()):,})", flush=True)
    for b in BANDS:
        s, p = stats[b], stats_pre[b]
        print(f"  {b:11s} n={s['n']:4d}  spikes {s['spikes']:8,d}  "
              f"{s['hz']:8.2f} Hz/neuron  silent {s['silent']}"
              f"   | untrained {p['hz']:8.2f} Hz  silent {p['silent']}", flush=True)

    # how much of the activity is locked to the stimulus period vs free-running: compare the
    # population rate's autocorrelation at the EPISODE lag against its off-period neighbours
    for tag, mat in (("trained", R), ("untrained", R_pre)):
        r = mat.sum(0).astype(np.float64)
        r = r - r.mean()
        if r.std() > 0:
            ac = np.array([np.corrcoef(r[:-l], r[l:])[0, 1] for l in range(1, 3 * EPISODE)])
            lags = [int(v) for v in np.argsort(ac)[::-1][:5] + 1]
            print(f"  population-rate autocorrelation ({tag}): peak lags {lags} "
                  f"(stimulus period is {EPISODE}); ac[{EPISODE}]={ac[EPISODE - 1]:+.3f}",
                  flush=True)

    plot(R, counts, stats, a, train_s, h["n_syn"])
    print(f"\nwrote {os.path.abspath(a.out)}", flush=True)


def plot(R, counts, stats, a, train_s, n_syn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.cumsum([0] + [counts[b] for b in BANDS])
    # Bands differ 130-fold in size (6 output vs 800 excitatory). One shared neuron axis would
    # render input and output as two hairlines, so each band gets its own panel, sized between
    # its true share and a readable floor -- the counts are on every label.
    heights = [max(0.9, min(4.6, counts[b] / 190.0)) for b in BANDS] + [1.15]
    onsets = np.arange(0, R.shape[1], EPISODE)
    fig, axes = plt.subplots(len(BANDS) + 1, 1, figsize=(13, sum(heights) + 2.2),
                             sharex=True, layout="constrained",
                             gridspec_kw=dict(height_ratios=heights))

    for ax, b, lo, hi in zip(axes, BANDS, edges[:-1], edges[1:]):
        sub = R[lo:hi]
        y, x = np.nonzero(sub)
        # stimulus onsets first, so they sit UNDER the spikes: structure locked to these lines
        # is the drive echoing through, structure between them is the net's own
        for t in onsets:
            ax.axvline(t, color="0.82", lw=0.7, zorder=0)
        ax.scatter(x, y, s=0.45, c=COLORS[b], marker="s", linewidths=0, rasterized=True,
                   zorder=2)
        ax.set_ylim(-0.5, sub.shape[0] - 0.5)
        ax.set_ylabel(f"{b}\nn={sub.shape[0]}", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        st = stats[b]
        ax.text(0.008, 0.94, f"{st['spikes']:,} spikes | {st['hz']:.1f} Hz/neuron",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.3",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5), zorder=3)

    # population rate: the rhythm question in one line
    rate = axes[-1]
    for t in onsets:
        rate.axvline(t, color="0.82", lw=0.7, zorder=0)
    rate.plot(R.sum(0), color="0.2", lw=0.8, zorder=2)
    rate.set_ylabel("all spikes\nper tick", fontsize=9)
    rate.tick_params(labelsize=8)
    for s in ("top", "right"):
        rate.spines[s].set_visible(False)
    rate.set_xlabel(f"tick (1 tick = 1 ms). Grey lines: dataset-state onsets, every "
                    f"{EPISODE} ticks", fontsize=10)
    rate.set_xlim(0, R.shape[1])

    axes[0].set_title(
        f"SPNet spike raster, {a.window}-tick window after {a.ticks:,} ticks of STDP on real "
        f"dataset input\n"
        f"{sum(counts.values())} neurons: "
        f"{', '.join(f'{counts[b]} {b}' for b in BANDS)}; {n_syn:,} synapses\n"
        f"stdp_lr={a.stdp_lr}, inhibitory frozen at {S.RES_W_INH}, trained "
        f"{train_s / 60:.0f} min on GPU; window captured with STDP off",
        fontsize=10.5, loc="left", pad=8)
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
