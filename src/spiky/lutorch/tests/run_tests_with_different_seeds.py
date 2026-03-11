"""
Aggregate runner for LUTorch tests.

This script delegates to the individual test modules' `main()` functions so
their own device/seed logic is reused:

- spiky.lut_fused.tests.test_anchor_sampler
- spiky.lutorch.tests.test_multi_head_lut
- spiky.lutorch.tests.test_lut_cross_attention
- spiky.lutorch.tests.test_gt_vs_lut_transformer
"""

from spiky.lutorch.tests.test_multi_head_lut import main as _run_multi_head_lut
from spiky.lutorch.tests.test_lut_cross_attention import main as _run_lut_cross_attention
from spiky.lutorch.tests.test_gt_vs_lut_transformer import main as _run_gt_vs_lut_transformer


def main() -> int:
    runners = [
        ("multi_head_lut", _run_multi_head_lut),
        ("lut_cross_attention", _run_lut_cross_attention),
        ("gt_vs_lut_transformer", _run_gt_vs_lut_transformer),
    ]

    for name, runner in runners:
        print("\n" + "=" * 60)
        print(f"Running LUTorch test suite: {name}")
        print("=" * 60)
        rc = runner()
        if rc not in (0, None):
            print(f"\n{name} tests failed with return code {rc}")
            return rc

    print("\nAll LUTorch test suites completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

