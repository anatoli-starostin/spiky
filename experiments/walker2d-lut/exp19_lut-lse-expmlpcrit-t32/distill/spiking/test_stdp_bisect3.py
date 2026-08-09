"""Third bisection: is the trigger per-neuron CAPACITY, not plasticity or meta count?

HYPOTHESIS. In explicit wiring, a neuron's outgoing synapses are split into groups BY
META. With one meta, 16 synapses pack into 2 groups of 8 = 16 slots. With 20 metas the
same 16 synapses scatter across up to 16 distinct metas, each needing its OWN group of 8
= up to 128 slots. So the meta count inflates the per-neuron slot requirement by up to the
group size, and `register_neuron_type(max_synapses=...)` is what bounds it. If that is the
story, raising max_synapses alone fixes 20 metas with no engine change at all.
"""
import subprocess
import sys

HERE = __file__.replace("test_stdp_bisect3.py", "test_stdp.py")
NE = 64

for n_meta in (1, 2, 4, 20):
    for mult in (1, 2, 4, 8):
        r = subprocess.run([sys.executable, HERE, "--case", "B",
                            "--n-meta", str(n_meta),
                            "--max-syn", str(NE * mult)],
                           capture_output=True, text=True)
        txt = r.stdout + r.stderr
        v = "PASS " if "PASS" in txt else "CRASH"
        n = [ln for ln in txt.splitlines() if "PASS" in ln]
        extra = f"  ({n[0].split('synapses')[1].split()[0]} synapses)" if n else ""
        print(f"  metas {n_meta:2d}  max_synapses {NE*mult:4d} ({mult}x) -> {v}{extra}",
              flush=True)
