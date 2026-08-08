"""Kendall tau-b (ties-aware) fitness for the LUT-ordering task — a DENSE, partial-credit
signal to replace the near-all-or-nothing strict+concordance fitness whose flat gradient
stalls evolution.

Ranking convention: an output is "better" when its target LUT value is HIGHER and, in the
prediction, when it fires EARLIER (smaller first-spike tick). So we compute tau-b between
u = target values and v = -first_spike (both higher==better; ties in first-spike -> ties in v,
ties in target values -> ties in u, both handled gracefully). tau-b in [-1,1] is normalized to
[0,1] via (tau_b+1)/2. Vectorized in torch over (…, Dout); a brute-force per-case reference
(taub_ref) is the correctness oracle (scipy is unavailable here).
"""
import torch

import neuroevo_lut as N
from lutmodel import lut_bits, bits_to_row

ORD_PAIRS = N._ORD_PAIRS      # the 6 output pairs (Dout=4)
Dout = N.Dout
N_TICKS = N.N_TICKS


def taub_vec(first, tvals):
    """first, tvals: tensors with a trailing Dout axis and broadcastable leading dims.
    Returns normalized tau-b in [0,1] over the leading dims (float32)."""
    v = -first.to(torch.float32)     # earlier spike -> higher rank
    u = tvals.to(torch.float32)      # higher LUT value -> higher rank
    C = D = Tu = Tv = 0.0
    for a, b in ORD_PAIRS:
        su = torch.sign(u[..., a] - u[..., b])
        sv = torch.sign(v[..., a] - v[..., b])
        prod = su * sv
        C = C + (prod > 0).float()
        D = D + (prod < 0).float()
        Tu = Tu + ((su == 0) & (sv != 0)).float()
        Tv = Tv + ((sv == 0) & (su != 0)).float()
    denom = torch.sqrt((C + D + Tu) * (C + D + Tv))
    taub = torch.where(denom > 0, (C - D) / denom.clamp(min=1e-12), torch.zeros_like(denom))
    return (taub + 1.0) / 2.0


def taub_ref(first_row, tvals_row):
    """Brute-force per-case tau-b (the scipy-equivalent 'b' variant). first_row/tvals_row are
    length-Dout sequences. Returns normalized tau-b in [0,1]."""
    v = [-float(f) for f in first_row]
    u = [float(t) for t in tvals_row]
    C = D = Tu = Tv = 0
    for a in range(len(u)):
        for b in range(a + 1, len(u)):
            su = (u[a] > u[b]) - (u[a] < u[b])
            sv = (v[a] > v[b]) - (v[a] < v[b])
            p = su * sv
            if p > 0:
                C += 1
            elif p < 0:
                D += 1
            elif su == 0 and sv != 0:
                Tu += 1
            elif sv == 0 and su != 0:
                Tv += 1
    denom = ((C + D + Tu) * (C + D + Tv)) ** 0.5
    taub = (C - D) / denom if denom > 0 else 0.0
    return (taub + 1.0) / 2.0


def run_equiv_test(n=4000, seed=0):
    """Vectorized taub_vec vs brute-force taub_ref on random AND heavily-tied cases."""
    g = torch.Generator().manual_seed(seed)
    # random cases: first-spikes in 1..N_TICKS (+ sentinel), target values continuous
    first = torch.randint(1, N_TICKS + 2, (n, Dout), generator=g).float()
    tvals = (torch.rand((n, Dout), generator=g) * 6 - 3)
    # tied cases: quantize to force many ties in both first-spikes and target values
    firstT = torch.randint(1, 4, (n, Dout), generator=g).float()          # only 3 distinct ticks -> ties
    tvalsT = torch.randint(-1, 2, (n, Dout), generator=g).float()          # only {-1,0,1} -> ties
    # a few all-equal degenerate rows
    firstD = torch.full((50, Dout), 7.0)
    tvalsD = torch.full((50, Dout), 2.0)
    first = torch.cat([first, firstT, firstD])
    tvals = torch.cat([tvals, tvalsT, tvalsD])
    vec = taub_vec(first, tvals)
    ref = torch.tensor([taub_ref(first[i].tolist(), tvals[i].tolist()) for i in range(first.shape[0])])
    maxerr = (vec - ref).abs().max().item()
    exact_match = torch.allclose(vec, ref, atol=1e-6)
    n_tied_cases = int(((first[:, :, None] == first[:, None, :]).sum(dim=(1, 2)) > Dout).sum())
    return maxerr, exact_match, first.shape[0], n_tied_cases


def fixed_eval_taub():
    """(xs, tvals, true_orders): tvals = the 4 LUT target values per input (ties possible);
    true_orders = oracle ranking (for the legacy strict/exact-order metric). Uses the SAME
    fixed eval set as evo_config.fixed_eval_set so results stay comparable."""
    from evo_config import fixed_eval_set
    xs, tos = fixed_eval_set()
    tvals = [list(N.m["V"][bits_to_row(lut_bits(N.m, x))]) for x in xs]
    return xs, tvals, tos


def score_population_taub(packed, ncand, xs, tvals, true_orders, blend=0.0, all_fire_gate=True):
    """Vectorized tau-b fitness for a packed population. Returns per-candidate
    (fitness, strict_frac, taub_mean) where fitness = (1-blend)*taub + blend*strict_exact.
    Reuses the same input construction / first-spike readout as N._score_population.
    all_fire_gate: if True, a corner with any silent output contributes 0 (keeps the
    all-outputs-fire constraint)."""
    import torch as T
    gid = packed["gid"]
    sp = packed["spnet"]
    dev = T.device("cpu")
    D = N.D
    nb = len(xs)
    in_gid = T.tensor([[gid(c, i) for i in range(D)] for c in range(ncand)], dtype=T.int32)
    Oids = T.tensor([gid(c, D + d) for c in range(ncand) for d in range(Dout)], dtype=T.int32)
    xs_t = T.tensor(xs, dtype=T.float32)
    ticks = (N.A_LAT - N.B_LAT * xs_t).round().long().clamp(1, N_TICKS - 1)
    kdim = ncand * D
    S = T.zeros((nb, N_TICKS, kdim), dtype=T.int32)
    Vv = T.zeros((nb, N_TICKS, kdim), dtype=T.float32)
    bb = T.arange(nb).view(nb, 1, 1).expand(nb, D, ncand)
    ii = T.arange(D).view(1, D, 1).expand(nb, D, ncand)
    cc = T.arange(ncand).view(1, 1, ncand).expand(nb, D, ncand)
    tt = ticks.view(nb, D, 1).expand(nb, D, ncand)
    col = cc * D + ii
    val = in_gid.t().reshape(1, D, ncand).expand(nb, D, ncand)
    S[bb.reshape(-1), tt.reshape(-1), col.reshape(-1)] = val.reshape(-1)
    Vv[bb.reshape(-1), tt.reshape(-1), col.reshape(-1)] = 50.0
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=nb, n_input_ticks=N_TICKS,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, N.NeuronDataType.Spike, 0, N_TICKS - 1).view(nb, ncand, Dout, N_TICKS)
    first = N._first_spike(spk, N_TICKS + 1)                                  # (nb, ncand, Dout)
    tvals_t = T.tensor(tvals, dtype=T.float32).view(nb, 1, Dout)              # (nb,1,Dout)
    taub = taub_vec(first, tvals_t)                                          # (nb, ncand)
    if all_fire_gate:
        fired = (first < N_TICKS).all(dim=2).float()
        taub = taub * fired
    strict, _ = N._strict_conc(first, true_orders)                           # (nb, ncand) exact-order match
    taub_mean = taub.mean(dim=0)
    sf = strict.mean(dim=0)
    fit = (1.0 - blend) * taub_mean + blend * sf
    return [(float(fit[c]), float(sf[c]), float(taub_mean[c])) for c in range(ncand)]


if __name__ == "__main__":
    maxerr, ok, ncases, ntied = run_equiv_test()
    print("tau-b equivalence test: %d cases (%d with ties) | max |vec-ref| = %.2e | within 1e-6: %s"
          % (ncases, ntied, maxerr, ok))
