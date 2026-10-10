"""Static source check (not a behavioural test): every CUDA kernel in cartridges/csrc widens blockIdx.x to int64
BEFORE multiplying by blockDim.x.

`int64_t i = blockIdx.x * blockDim.x + threadIdx.x` multiplies in 32-bit unsigned and only then widens, so past 2^32
threads the id wraps, passes the `i < total` bounds check and silently writes the wrong elements. That wrap needs
> 4.29e9 threads (for softsign_surrogate_grad: > 4.29e9 per-table entries, ~100 GB of inputs) and cannot be exercised
on available hardware, so this checks the source pattern instead, the way every kernel here is written:
`static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x` (or `(int64_t)blockIdx.x * ...`).
"""
import re
from pathlib import Path

CSRC = Path(__file__).resolve().parents[1] / "cartridges" / "csrc"
WIDENED = re.compile(r"(static_cast<\s*int64_t\s*>\s*\(\s*blockIdx\.x\s*\)|\(\s*int64_t\s*\)\s*blockIdx\.x)")


def test_every_thread_index_is_widened_before_the_multiply():
    sources = sorted(CSRC.glob("*.cu")) + sorted(CSRC.glob("*.cuh"))
    assert sources, f"no CUDA sources found under {CSRC}"
    offenders, seen = [], 0
    for src in sources:
        for n, line in enumerate(src.read_text().splitlines(), 1):
            code = line.split("//", 1)[0]
            if "blockIdx.x" in code and "*" in code:
                seen += 1
                if not WIDENED.search(code):
                    offenders.append(f"{src.name}:{n}: {line.strip()}")
    assert seen >= 20, f"expected the ~22 kernel thread-id lines, found {seen}"
    assert not offenders, "32-bit thread-id products (widen blockIdx.x first):\n" + "\n".join(offenders)
