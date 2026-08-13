"""Gate set for arm B importing the matured 3600 s reservoir artifact.

Reservoir-edge accounting, correctly this time: a reservoir edge is one whose SOURCE **and**
TARGET are both reservoir neurons. Testing the source alone counts the output synapses
(sourced from excitatory cells) as reservoir edges and overshoots.
"""
import argparse
import os

import numpy as np
import torch

import es_harness as H
from es_harness import (LatencyEncoder, N_EXC, N_TICKS, build, kendall_tau_b,
                        random_genome, reservoir_wiring, run_episode, verify_round_trip)
from es_smoke import load

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "results", "reservoir_b_3600s.npz"))
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Z = np.load(a.npz, allow_pickle=True)
    edges = (Z["src_local"], Z["src_is_inh"], Z["tgt_local"], Z["tgt_is_inh"],
             Z["weight"], Z["delay"])
    n_cap = edges[0].shape[0]
    print(f"artifact {os.path.basename(a.npz)}: {n_cap:,} synapses, "
          f"{int(Z['stdp_seconds'])} s STDP, seed {int(Z['seed'])}")
    we = Z["weight"][~Z["src_is_inh"]]
    print(f"  exc weights mean {we.mean():.3f} sd {we.std():.3f} [{we.min():.2f}, {we.max():.2f}]"
          f"  at~0 {float((we <= 0.1).mean()):.4f}  at~max {float((we >= 9.9).mean()):.4f}")

    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, a.seed)
    enc = LatencyEncoder(Xpool)
    res = reservoir_wiring(np.random.default_rng(1234))
    rng = np.random.default_rng(a.seed)
    g0 = random_genome(rng, a.w_max)

    print("\n=== ROUND TRIP (imported 3600 s reservoir) ===")
    for d in (["cuda", "cpu"] if dev == "cuda" else ["cpu"]):
        h = build([g0], res, d, res_edges=edges)
        r = verify_round_trip(h)
        E = r["n_requested"]
        ids, tri = h["ids"], h["triples"]
        pool = np.concatenate([ids[0], ids[1]])
        is_res = np.isin(tri[:, 1], pool) & np.isin(tri[:, 2], pool)   # BOTH ends
        n_res = int(is_res.sum())
        ok = r["weights_ok"] == E and r["delays_ok"] == E and r["missing"] == 0
        print(f"  {d:4s}: captured {n_cap:,} | regrown {E:,} "
              f"(reservoir {n_res:,} + I/O {E - n_res:,}) | weights {r['weights_ok']}/{E} "
              f"delays {r['delays_ok']}/{E} missing {r['missing']} dropped {n_cap - n_res} "
              f"-> {'EDGE-EXACT' if ok and n_cap == n_res else 'MISMATCH'}")
        if d == dev:
            h1 = h
        else:
            del h

    print("\n=== SINGLE-CANDIDATE SMOKE ===")
    from spiky.spnet.spnet import NeuronDataType
    sp, ids = h1["spnet"], h1["ids"]
    first, _ = run_episode(h1, X, enc, a.current)
    tk = enc(X)
    Rin = sp.export_neuron_data(torch.tensor(ids[2], dtype=torch.int32, device=dev),
                                X.shape[0], NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    Rexc = sp.export_neuron_data(torch.tensor(ids[0], dtype=torch.int32, device=dev),
                                 X.shape[0], NeuronDataType.Spike, 0, N_TICKS - 1).cpu().numpy()
    fin = Rin.any(-1)
    ft = np.where(fin, Rin.argmax(-1), -1)
    of = first < N_TICKS
    print(f"  input neurons fired {fin.mean()*100:.1f}%, median lag "
          f"{np.median((ft - tk)[fin]):.1f} ticks")
    print(f"  reservoir {Rexc.sum()/X.shape[0]:.1f} spikes/sample, "
          f"{Rexc.any(-1).sum()/X.shape[0]/N_EXC*100:.1f}% exc recruited")
    print(f"  outputs fired {of.mean()*100:.1f}%, first-spike ticks "
          f"{first[of].min():.0f}-{first[of].max():.0f}")

    print(f"\n=== ISOLATION, {a.pop} packed vs solo ===")
    gs = [g0] + [random_genome(rng, a.w_max) for _ in range(a.pop - 1)]
    hp = build(gs, res, dev, res_edges=edges)
    _, Rp = run_episode(hp, X, enc, a.current)
    allok = True
    for c in range(a.pop):
        hs = build([gs[c]], res, dev, res_edges=edges)
        _, Rs = run_episode(hs, X, enc, a.current)
        ok = np.array_equal(Rp[:, c], Rs[:, 0])
        print(f"  candidate {c}: raster identical {ok} "
              f"(differing {int((Rp[:, c] != Rs[:, 0]).sum())})")
        allok &= ok
        del hs
    print(f"  ISOLATION GATE: {'PASS' if allok else 'FAIL'}")

    print("\n=== NULL BASELINE (200 label shuffles, fixed reference batch) ===")
    tau = kendall_tau_b(-first[:, 0, :], Y)
    nulls = np.array([kendall_tau_b(-first[:, 0, :],
                                    Y[np.random.default_rng(k).permutation(Y.shape[0])]).mean()
                      for k in range(200)])
    print(f"  random-wiring tau-b  {tau.mean():+.4f}  (SE {tau.std()/np.sqrt(len(tau)):.4f})")
    print(f"  label-shuffle null   {nulls.mean():+.4f}  sd {nulls.std():.4f}  "
          f"[{np.percentile(nulls,2.5):+.4f}, {np.percentile(nulls,97.5):+.4f}]")
    print(f"  => {(tau.mean()-nulls.mean())/max(nulls.std(),1e-9):+.2f} sd above null   "
          f"(tau - null = {tau.mean()-nulls.mean():+.4f})")


if __name__ == "__main__":
    main()
