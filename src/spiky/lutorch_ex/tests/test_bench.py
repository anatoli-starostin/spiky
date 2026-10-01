"""Smoke tests for the benchmark harness: the CPU grid runs and rows are well-formed."""
import json

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT
from spiky.lutorch_ex.bench import (
    BenchRow,
    benchmark,
    format_table,
    rows_to_csv,
    rows_to_json,
)


def _make():
    spec = LUTSpec(h_in=1, h_out=1, tph=2, nap=3, d_in=4, d_out=4)
    return ManifestoHardLUT(spec, seed=0)


def test_cpu_grid_runs_and_is_well_formed():
    ops = ("forward_eval", "forward_train", "backward")
    batches = (1, 8)
    devices = ("cpu", "cuda:H100")  # cuda stays a placeholder (measure_cuda defaults False)
    rows = benchmark(_make, name="hard", batch_sizes=batches, operations=ops,
                     devices=devices, warmup=1, repeats=2)
    assert len(rows) == len(ops) * len(batches) * len(devices)
    assert all(isinstance(r, BenchRow) for r in rows)

    cpu = [r for r in rows if r.device == "cpu"]
    assert len(cpu) == len(ops) * len(batches)
    for r in cpu:
        assert r.status == "ok", r.note
        assert r.median_ms is not None and r.median_ms > 0
        assert r.min_ms is not None and r.min_ms <= r.median_ms
        assert r.throughput_rows_per_s is not None and r.throughput_rows_per_s > 0

    gpu = [r for r in rows if r.device == "cuda:H100"]
    assert gpu and all(r.status == "placeholder" and r.note for r in gpu)


def test_outputs_serialise():
    rows = benchmark(_make, name="hard", batch_sizes=(1,), operations=("forward_eval",),
                     devices=("cpu",), warmup=1, repeats=2)
    assert isinstance(format_table(rows), str) and "forward_eval" in format_table(rows)
    parsed = json.loads(rows_to_json(rows))
    assert parsed and parsed[0]["operation"] == "forward_eval" and parsed[0]["status"] == "ok"
    csv_text = rows_to_csv(rows)
    assert "cartridge" in csv_text.splitlines()[0]
