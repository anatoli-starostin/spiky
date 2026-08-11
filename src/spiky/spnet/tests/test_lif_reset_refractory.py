#!/usr/bin/env python3
"""Tests for the per-neuron reset mode (constant/subtractive) + optional refractory period,
and the LIFNeuronMeta wrapper (GitHub issue #96). Self-contained; runs on the CPU build.

Covers:
  (a) default Izhikevich reset is unchanged (constant reset to c) — backward compatibility,
  (b) constant-reset LIF fires and resets to v_reset,
  (c) subtractive-reset carries the threshold overshoot (post-spike V higher than constant),
  (d) an absolute refractory period blocks spikes for the configured window.
"""
import torch

from spiky.spnet.spnet import SpikingNet, NeuronMeta, LIFNeuronMeta, SynapseMeta, NeuronDataType
from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.util.test_utils import grow_and_add


def _build(neuron_metas, counts, device='cpu', dtype=torch.float32):
    """Minimal net: the given neuron metas + one dummy (weight-0) synapse so compile has work.
    External current is injected directly, so the (unused) synapse doesn't affect the dynamics."""
    sms = [SynapseMeta(learning_rate=0.0, min_delay=0, max_delay=0, initial_weight=0.0,
                       _forward_group_size=8, _backward_group_size=8)]
    ge = SynapseGrowthEngine(device=device, synapse_group_size=8, max_groups_in_buffer=256)
    for _ in neuron_metas:
        ge.register_neuron_type(max_synapses=64, growth_command_list=[])
    spnet = SpikingNet(synapse_metas=sms, neuron_metas=neuron_metas, neuron_counts=counts,
                       summation_dtype=dtype)
    spnet.to_device(device)
    for mi, cnt in enumerate(counts):
        ids = spnet.get_neuron_ids_by_meta(mi)
        coords = torch.stack([torch.arange(cnt).float(), torch.zeros(cnt), torch.zeros(cnt)], dim=1)
        ge.add_neurons(neuron_type_index=mi, identifiers=ids, coordinates=coords)
    ids0 = spnet.get_neuron_ids_by_meta(0)
    triples = torch.tensor([[0, int(ids0[0]), int(ids0[1])]], dtype=torch.int32, device=device)
    grow_and_add(ge, spnet, 1, 1, explicit_triples=triples)
    spnet.compile(shuffle_synapses_random_seed=None)
    return spnet


def _run(spnet, inject_ids, current, n_ticks, do_train=False):
    input_values = torch.full([1, inject_ids.shape[0], n_ticks], float(current))
    ns = spnet.process_ticks(n_ticks_to_process=n_ticks, batch_size=1, n_input_ticks=n_ticks,
                             input_values=input_values, do_train=do_train,
                             input_neuron_ids=inject_ids, do_record_voltage=True)
    V = spnet.export_neuron_data(inject_ids, 1, NeuronDataType.Voltage, 0, n_ticks - 1)
    S = spnet.export_neuron_data(inject_ids, 1, NeuronDataType.Spike, 0, n_ticks - 1)
    return ns, V[0], S[0]


def _spike_ticks(S_row):
    return torch.where(S_row > 0)[0].tolist()


def test_a_izhikevich_unchanged():
    """Default Izhikevich neuron: fires and does the CONSTANT hard reset to c (=-65), not subtractive."""
    spnet = _build([NeuronMeta(neuron_type=0)], [8])  # all defaults: reset_mode=0, refractory=0
    ids = spnet.get_neuron_ids_by_meta(0)[:4]
    _, V, S = _run(spnet, ids, 12.0, 80)
    st = _spike_ticks(S[0])
    assert len(st) >= 2, f"default Izhikevich should spike under drive; got {st}"
    # tick after a spike: V must be near c=-65 (constant reset), NOT near 0 (which subtractive would give)
    after = V[0, st[0] + 1].item()
    assert after < -40.0, f"Izhikevich reset should be constant to c(-65); post-spike V={after}"
    print(f"  (a) Izhikevich unchanged: spikes at {st[:5]}, post-spike V={after:.2f} (constant reset to c) OK")
    return True


def test_b_constant_lif():
    """Constant-reset LIF fires and resets to v_reset (regular firing)."""
    lif = LIFNeuronMeta(neuron_type=0, tau=20.0, v_rest=0.0, v_reset=0.0, threshold=1.0)
    spnet = _build([lif], [8])
    ids = spnet.get_neuron_ids_by_meta(0)[:4]
    _, V, S = _run(spnet, ids, 0.12, 60)
    st = _spike_ticks(S[0])
    assert len(st) >= 3, f"constant LIF should fire repeatedly; got {st}"
    after = V[0, st[0] + 1].item()
    assert abs(after) < 0.25, f"post-spike V should be near v_reset(0)+one-step charge; got {after}"
    isis = [st[i + 1] - st[i] for i in range(len(st) - 1)]
    assert max(isis) - min(isis) <= 1, f"constant LIF should fire regularly; ISIs={isis}"
    print(f"  (b) constant LIF: spikes at {st}, ISIs={isis}, post-spike V={after:.3f} OK")
    return True


def test_c_subtractive_carries_overshoot():
    """Subtractive reset carries the threshold overshoot: post-spike V is higher than constant reset."""
    common = dict(neuron_type=0, tau=20.0, v_rest=0.0, v_reset=0.0, threshold=1.0)
    sp_const = _build([LIFNeuronMeta(**common, subtractive_reset=False)], [8])
    sp_sub = _build([LIFNeuronMeta(**common, subtractive_reset=True)], [8])
    idc = sp_const.get_neuron_ids_by_meta(0)[:4]
    ids = sp_sub.get_neuron_ids_by_meta(0)[:4]
    _, Vc, Sc = _run(sp_const, idc, 0.12, 60)
    _, Vs, Ss = _run(sp_sub, ids, 0.12, 60)
    stc, sts = _spike_ticks(Sc[0]), _spike_ticks(Ss[0])
    assert stc and sts, f"both should spike; const={stc}, sub={sts}"
    v_over = Vs[0, sts[0]].item()                       # the overshoot value at the spike tick (>threshold)
    after_c = Vc[0, stc[0] + 1].item()                  # constant: v_reset + charge
    after_s = Vs[0, sts[0] + 1].item()                  # subtractive: (v_over - threshold) + charge
    assert v_over > 1.0, f"expected an overshoot above threshold; got {v_over}"
    assert after_s > after_c + 0.005, (
        f"subtractive should carry overshoot: after_sub={after_s:.4f} vs after_const={after_c:.4f}")
    print(f"  (c) subtractive: overshoot@spike={v_over:.4f}, post-spike sub={after_s:.4f} > const={after_c:.4f} OK")
    return True


def test_d_refractory_blocks():
    """An absolute refractory period enforces a minimum inter-spike interval."""
    R = 12
    # strong drive so, without refractory, it would fire fast; refractory must space spikes >= R.
    lif = LIFNeuronMeta(neuron_type=0, tau=20.0, v_rest=0.0, v_reset=0.0, threshold=1.0, refractory_ticks=R)
    ref = _build([lif], [8])
    idr = ref.get_neuron_ids_by_meta(0)[:4]
    # baseline (same params, no refractory) to confirm it WOULD fire faster than R
    base = _build([LIFNeuronMeta(neuron_type=0, tau=20.0, v_rest=0.0, v_reset=0.0, threshold=1.0)], [8])
    idb = base.get_neuron_ids_by_meta(0)[:4]
    _, _, Sb = _run(base, idb, 0.30, 90, do_train=True)
    _, _, Sr = _run(ref, idr, 0.30, 90, do_train=True)
    stb, str = _spike_ticks(Sb[0]), _spike_ticks(Sr[0])
    isib = [stb[i + 1] - stb[i] for i in range(len(stb) - 1)]
    isir = [str[i + 1] - str[i] for i in range(len(str) - 1)]
    assert isib and min(isib) < R, f"baseline should fire faster than refractory R={R}; ISIs={isib}"
    assert isir and min(isir) >= R, f"refractory must enforce ISI >= R={R}; got ISIs={isir}"
    print(f"  (d) refractory R={R}: baseline ISIs={isib} (min<{R}), refractory ISIs={isir} (min>={R}) OK")
    return True


def main():
    ok = True
    print("LIF reset-mode + refractory tests (CPU):")
    for t in (test_a_izhikevich_unchanged, test_b_constant_lif,
              test_c_subtractive_carries_overshoot, test_d_refractory_blocks):
        try:
            ok = t() and ok
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
            ok = False
    print("🎉 all passed" if ok else "❌ FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
