"""lutorch_ex benchmarks — the benchmark scripts.

* ``bench`` — the benchmark harness (``benchmark()``, ``format_table``, the JSON / CSV export).
* ``bench_snapshot`` — a performance snapshot of every class (one fresh process per cell).
* ``bench_compare`` — two snapshot JSONs -> the comparison figures.

Run them as modules, e.g. ``python -m spiky.lutorch_ex.benchmarks.bench_snapshot``.
Nothing is imported eagerly: ``bench_compare`` needs matplotlib, which the library does not.
"""
