"""exp_c31 — full teardown of the seed-0 PureLIF actor (CPU-ref 4262.1).

Every trainable tensor is compared against ITS OWN INIT, not against a fresh random draw:
`pure_lif_sac.py` derives the actor key as `split(PRNGKey(seed), 4)[1]`, so the exact
starting point is reproducible and "how far did this move" is a real measurement rather
than a distributional guess.

States come from the DEPLOYED policy (mode="hard", tanh(mu)) rolled out in MJX, not from
the replay buffer: the question is what this actor does in the states it actually visits,
and the buffer is contaminated with 500 iterations of uniform-random warmup plus every
policy the run passed through.

Writes analysis_seed0.npz for plot_analysis.py (matplotlib lives in the other venv) and
prints the written report to stdout.

Usage:
  python analyze_seed0.py [--states 4096] [--episodes 24]
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W                                   # noqa: E402
from mujoco import mjx                                     # noqa: E402
import jax_pure_lif as LIF                                 # noqa: E402

OBS, ACT = 17, 6
NAP, TPH, HEADS = 6, 32, 1
N_TABLES, M = HEADS * TPH, HEADS * TPH * NAP

# Walker2d-v5: obs = qpos[1:] (8) ++ clip(qvel, -10, 10) (9).
CH = ["z height", "torso ang", "R thigh ang", "R leg ang", "R foot ang",
      "L thigh ang", "L leg ang", "L foot ang",
      "vx", "vz", "torso angvel", "R thigh vel", "R leg vel", "R foot vel",
      "L thigh vel", "L leg vel", "L foot vel"]
PARAM_KEYS = ("delay", "w", "L", "tau_raw", "log_T_cross", "log_temp_bit", "table")


def load():
    z = np.load(os.path.join(HERE, "pure_lif_sac_c31_s0_actor.npz"))
    fin = {k: jnp.asarray(z[k]) for k in PARAM_KEYS}
    # Reproduce the init exactly as the trainer built it.
    key = jax.random.PRNGKey(0)
    _, ka, _, _ = jax.random.split(key, 4)
    ini = LIF.init(ka, NAP, TPH, HEADS, OBS, 2 * ACT)
    ini["table"] = ini["table"].at[:, :, ACT:].add(-1.0 / (HEADS * TPH))
    return fin, ini


def rollout_states(n_states, episodes, horizon=1000):
    """Observations visited by the DEPLOYED policy."""
    fin, _ = load()
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    m = W.make_model()
    reset, step = W.make_env(mjx.put_model(m))
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(jax.random.PRNGKey(0), episodes))

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = LIF.apply(fin, x, 0.3, HEADS, TPH, NAP, mode="hard").sum(1)
        return jnp.tanh(y[:, :ACT])

    @jax.jit
    def run(st):
        def one(c, _):
            st, alive = c
            a = act(st.obs)
            nst = v_step(st, a)
            return (nst, alive * (1 - nst.done)), (st.obs, alive)
        (st, _), (obs, alive) = jax.lax.scan(
            one, (st, jnp.ones(episodes)), None, length=horizon)
        return obs, alive

    obs, alive = run(st)                                   # [T, E, 17], [T, E]
    obs = np.asarray(obs).reshape(-1, OBS)
    alive = np.asarray(alive).reshape(-1) > 0
    obs = obs[alive]                                       # drop post-termination padding
    idx = np.linspace(0, len(obs) - 1, min(n_states, len(obs))).astype(int)
    return jnp.asarray(obs[idx]), (om, osd), len(obs)


def chunked(fn, x, chunk=512):
    return np.concatenate([np.asarray(fn(x[i:i + chunk]))
                           for i in range(0, x.shape[0], chunk)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=4096)
    ap.add_argument("--episodes", type=int, default=24)
    a = ap.parse_args()

    fin, ini = load()
    obs, (om, osd), n_visited = rollout_states(a.states, a.episodes)
    x = (obs - om) / (osd + 1e-6)
    out = {}
    P = print

    P(f"=== exp_c31 seed 0 — parameter teardown ===")
    P(f"states: {x.shape[0]:,} sampled from {n_visited:,} visited by the deployed policy "
      f"({a.episodes} episodes)\n")

    # ---------------------------------------------------------------- table
    t_f, t_i = np.asarray(fin["table"]), np.asarray(ini["table"])
    d = t_f - t_i
    mu_f, sg_f = t_f[:, :, :ACT], t_f[:, :, ACT:]
    P("--- TABLE (32 tables x 64 rows x 12 outputs = 24,576) ---")
    P(f"  init   std {t_i.std():.4f}  range [{t_i.min():+.3f}, {t_i.max():+.3f}]")
    P(f"  final  std {t_f.std():.4f}  range [{t_f.min():+.3f}, {t_f.max():+.3f}]")
    P(f"  |delta| mean {np.abs(d).mean():.4f}  max {np.abs(d).max():.3f}")
    for thr in (0.01, 0.1, 0.5):
        P(f"    entries moved > {thr:<5}: {100*(np.abs(d) > thr).mean():5.1f}%")
    P(f"  mu half   (6 outputs): std {mu_f.std():.4f} range "
      f"[{mu_f.min():+.3f}, {mu_f.max():+.3f}]")
    P(f"  sigma half(6 outputs): std {sg_f.std():.4f} range "
      f"[{sg_f.min():+.3f}, {sg_f.max():+.3f}]  (init biased -1/32 = -0.031)")
    # tanh saturation of the mu head, summed over the 32 tables
    mu_sum = mu_f.sum(0)                                   # if all tables addressed row r
    P(f"  |sum of mu over tables| would saturate tanh (>2.0) for "
      f"{100*(np.abs(mu_sum) > 2.0).mean():.1f}% of (row, action) pairs")

    # ------------------------------------------------- addressing, deployed
    addr = chunked(lambda xx: LIF.address(fin, xx, 0.3, HEADS, TPH, NAP), x)
    hard, soft, t_hard = [], [], []
    for i in range(0, x.shape[0], 512):
        hb, sb, th = LIF.spike_bits(fin, x[i:i + 512], N_TABLES, NAP)
        hard.append(np.asarray(hb)); soft.append(np.asarray(sb))
        t_hard.append(np.asarray(th))
    hard = np.concatenate(hard); t_hard = np.concatenate(t_hard)

    visit = np.zeros((N_TABLES, 1 << NAP), np.int64)
    for t in range(N_TABLES):
        visit[t] = np.bincount(addr[:, t], minlength=1 << NAP)
    used = (visit > 0).sum(1)
    p = visit / visit.sum(1, keepdims=True)
    ent = -(np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0)).sum(1)
    P(f"\n--- ADDRESSING, on states the deployed policy actually visits ---")
    P(f"  rows reached: {100*(visit > 0).mean():.1f}% of 2,048  "
      f"(training-cumulative coverage was 78.4%)")
    P(f"  distinct rows per table: min {used.min()} median "
      f"{int(np.median(used))} max {used.max()} of 64")
    P(f"  address entropy per table: mean {ent.mean():.2f} bits of a possible 6.00")
    top = visit.max(1) / visit.sum(1)
    P(f"  most-used row's share: mean {100*top.mean():.1f}%  "
      f"(median {100*np.median(top):.1f}%)")
    P(f"  bits set: {100*hard.mean():.1f}%  (training log said 39.3%)")
    bit_occ = hard.reshape(-1, N_TABLES, NAP).mean((0, 1))
    P(f"  per-bit-position occupancy: "
      f"{', '.join(f'{100*b:.0f}%' for b in bit_occ)}")
    det_occ = hard.reshape(-1, M).mean(0)
    P(f"  detectors never firing: {(det_occ == 0).sum()} of {M};  "
      f"always firing: {(det_occ == 1).sum()}")

    # ---------------------------------------------------------------- delay
    dl_f = np.asarray(fin["delay"])
    P(f"\n--- DELAY (192 detectors x 17 channels), init ALL ZERO ---")
    P(f"  final  mean {dl_f.mean():+.4f}  std {dl_f.std():.4f}  "
      f"range [{dl_f.min():+.3f}, {dl_f.max():+.3f}]")
    P(f"  latency spread of the inputs themselves: "
      f"t = clip(16 - 3x, 0, 32), std over visited states "
      f"{np.asarray(jnp.clip(16.0 - 3.0 * x, 0, 32)).std():.3f}")

    # Does the learned delay actually reorder arrivals? Compare the pairwise order of
    # a = latency + delay against latency alone (which is the delay=0 order).
    lat = np.asarray(jnp.clip(16.0 - 3.0 * x, 0.0, 32.0))          # [S, 17]
    sub = lat[:512]
    a_l = sub[:, None, :] + dl_f[None]                             # [s, M, 17]
    s_l = np.sign(a_l[:, :, :, None] - a_l[:, :, None, :])
    s_0 = np.sign(sub[:, None, :, None] - sub[:, None, None, :])
    iu = np.triu_indices(OBS, 1)
    disagree = (s_l[:, :, iu[0], iu[1]] != s_0[:, :, iu[0], iu[1]]).mean()
    same_perm = (np.argsort(a_l, -1) ==
                 np.argsort(np.broadcast_to(sub[:, None, :], a_l.shape), -1)
                 ).all(-1).mean()
    P(f"  pairwise arrival orders flipped vs delay=0: {100*disagree:.1f}% of "
      f"{OBS*(OBS-1)//2} pairs")
    P(f"  detectors whose full sort order is UNCHANGED from delay=0: "
      f"{100*same_perm:.1f}%")
    dl_ch = dl_f.mean(0)
    order = np.argsort(dl_ch)
    P(f"  pushed EARLIEST (most negative mean delay):")
    for i in order[:4]:
        P(f"      {CH[i]:<14} {dl_ch[i]:+.3f}")
    P(f"  pushed LATEST:")
    for i in order[-4:][::-1]:
        P(f"      {CH[i]:<14} {dl_ch[i]:+.3f}")

    # ------------------------------------------------------------------- w
    w_f, w_i = np.asarray(fin["w"]), np.asarray(ini["w"])
    P(f"\n--- w (192 x 17 synaptic drive), init 0.2*randn ---")
    P(f"  init  std {w_i.std():.4f}   final std {w_f.std():.4f}  "
      f"({w_f.std()/w_i.std():.2f}x)")
    P(f"  range [{w_f.min():+.3f}, {w_f.max():+.3f}]   "
      f"|delta| mean {np.abs(w_f - w_i).mean():.4f}")
    w_ch = np.abs(w_f).mean(0)
    o = np.argsort(-w_ch)
    P(f"  channels by mean |w| (the drive the detectors key off):")
    for i in o[:6]:
        P(f"      {CH[i]:<14} {w_ch[i]:.4f}   (init {np.abs(w_i).mean(0)[i]:.4f})")
    P(f"    ... weakest: " + ", ".join(f"{CH[i]} {w_ch[i]:.3f}" for i in o[-3:]))

    # ------------------------------------------------------- L, tau, T, temp
    L_f, L_i = np.asarray(fin["L"]), np.asarray(ini["L"])
    tau_f = np.asarray(jax.nn.softplus(fin["tau_raw"]) + 1e-3)
    tau_i = np.asarray(jax.nn.softplus(ini["tau_raw"]) + 1e-3)
    tc_f = np.exp(np.asarray(fin["log_T_cross"]))
    tb_f = np.exp(np.asarray(fin["log_temp_bit"]))
    P(f"\n--- DEADLINE L (192), init 16.000 = 0.5*t_window ---")
    P(f"  final mean {L_f.mean():.3f}  std {L_f.std():.3f}  "
      f"range [{L_f.min():.3f}, {L_f.max():.3f}]")
    P(f"  moved |delta| mean {np.abs(L_f - L_i).mean():.3f}  "
      f"max {np.abs(L_f - L_i).max():.3f}")
    P(f"  fraction of first-spike times BELOW the deadline (= bit set): "
      f"{100*(t_hard < L_f[None]).mean():.1f}%")
    P(f"  t_hard: {100*np.mean(t_hard >= 32.0):.1f}% of detector/state pairs never "
      f"fire (t = t_window)")
    P(f"\n--- PER-LUT time constants (32 each) ---")
    P(f"  tau      init {tau_i.mean():.4f} -> final mean {tau_f.mean():.4f} "
      f"[{tau_f.min():.3f}, {tau_f.max():.3f}]")
    P(f"  T_cross  init 1.0000 -> final mean {tc_f.mean():.4f} "
      f"[{tc_f.min():.4f}, {tc_f.max():.4f}]")
    P(f"  temp_bit init 1.0000 -> final mean {tb_f.mean():.6f} "
      f"[{tb_f.min():.6f}, {tb_f.max():.6f}]   ({1.0/tb_f.mean():.0f}x sharper)")

    # ------------------------------- does a channel's arrival reach the decision?
    # The tempting story after seeing panels 4 and 5 is "vx is pushed last AND has the
    # biggest weight, so it casts the decisive final vote". The opposite reading is
    # equally available -- it arrives after the crossing has already happened and is
    # ignored. This measures which is true, and it is the second one.
    fires = t_hard < 32.0
    a_arr = lat[:, None, :] + dl_f[None]                           # [S, M, 17]
    before = (a_arr <= t_hard[:, :, None]) & fires[:, :, None]
    reach = before.sum((0, 1)) / max(fires.sum(), 1)
    P(f"\n--- DOES EACH CHANNEL REACH THE DECISION? ---")
    P(f"  {100*fires.mean():.1f}% of (state, detector) pairs fire at all")
    P(f"  % of FIRING pairs where the channel arrives at or before the spike:")
    for i in np.argsort(-reach):
        P(f"      {CH[i]:<14} {100*reach[i]:5.1f}%   (mean |w| {w_ch[i]:.3f})")
    iv = CH.index("vx")
    P(f"  --> vx: largest |w| in the bank ({w_ch[iv]:.3f}) but reaches the decision in "
      f"only {100*reach[iv]:.1f}% of firing pairs (mean {100*reach.mean():.1f}%).")
    P(f"      vx mean arrival {a_arr[:, :, iv].mean():.2f} vs mean spike time "
      f"{t_hard[fires].mean():.2f}, deadline {L_f.mean():.2f}. Its weight is "
      f"largely INERT in the deployed hard readout.")

    # ------------------------------------------------- specialisation / gait
    # How often does each detector's bit TOGGLE along a trajectory? A detector that
    # flips at gait frequency is a phase unit; one that never flips is a constant.
    tog = (hard.reshape(-1, M)[1:] != hard.reshape(-1, M)[:-1]).mean(0)
    P(f"\n--- SPECIALISATION ---")
    P(f"  bit toggle rate per step: mean {100*tog.mean():.2f}%  "
      f"max {100*tog.max():.2f}%")
    P(f"  detectors constant across ALL sampled states: "
      f"{int((tog == 0).sum())} of {M} "
      f"({100*(tog == 0).mean():.0f}%)")
    live = tog > 0
    P(f"  of the {int(live.sum())} live detectors, mean occupancy "
      f"{100*det_occ[live].mean():.1f}%")
    # Which channel best explains each live detector's bit (point-biserial |r|).
    flat = hard.reshape(-1, M)
    xs = np.asarray(x)
    xz = (xs - xs.mean(0)) / (xs.std(0) + 1e-8)
    best_ch, best_r = np.zeros(M, int), np.zeros(M)
    for m_ in np.where(live)[0]:
        b = flat[:, m_]
        bz = (b - b.mean()) / (b.std() + 1e-8)
        r = np.abs((xz * bz[:, None]).mean(0))
        best_ch[m_], best_r[m_] = int(np.argmax(r)), float(r.max())
    cnt = np.bincount(best_ch[live], minlength=OBS)
    o2 = np.argsort(-cnt)
    P(f"  channel each live detector correlates with most:")
    for i in o2[:6]:
        if cnt[i]:
            P(f"      {CH[i]:<14} {cnt[i]:3d} detectors  "
              f"(mean |r| {best_r[live][best_ch[live] == i].mean():.2f})")

    np.savez(os.path.join(HERE, "analysis_seed0.npz"),
             table_final=t_f, table_init=t_i, delay=dl_f, w_final=w_f, w_init=w_i,
             L_final=L_f, L_init=L_i, tau=tau_f, t_cross=tc_f, temp_bit=tb_f,
             visit=visit, bit_occ=bit_occ, det_occ=det_occ, toggle=tog,
             delay_ch=dl_ch, w_ch=w_ch, best_ch=best_ch, best_r=best_r, reach=reach,
             fire_rate=np.float32(fires.mean()),
             live=live, t_hard_sample=t_hard[:512], channels=np.array(CH),
             disagree=np.float32(disagree), same_perm=np.float32(same_perm))
    P(f"\nwrote analysis_seed0.npz")


if __name__ == "__main__":
    main()
