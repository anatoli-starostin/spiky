"""lutorch_ex benchmarks — the benchmark scripts and their records.

* ``bench`` — the benchmark harness (``benchmark()``, ``format_table``, the JSON / CSV export).
* ``bench_snapshot`` — the performance snapshot of record (every class, one fresh process per cell).
* ``bench_compare`` — two snapshot JSONs -> the comparison figures.
* ``bench_fused_confidence`` — FusedConfidenceLUT vs compiled ConfidenceLUT timings and launch-knob sweeps.

Run them as modules, e.g. ``python -m spiky.lutorch_ex.benchmarks.bench_snapshot``. The snapshot JSON / .md
files, the comparison figure directories and the running note (``fused_quantised_stage0.md``) live here too.
Nothing is imported eagerly: ``bench_compare`` needs matplotlib, which the library does not.
"""
