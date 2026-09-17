"""KERNEL VALIDATION GATE -- run on the machine that will train, BEFORE preflight / launch.

    set -a; . ./run.env; set +a; unset SPIKY_P2_CUDA_DISABLE; "$PYTHON" validate_kernel.py

1. Builds the p2_int8 CUDA extension for THIS GPU (pow2_int8.load(); refuses if the capability is not in VALIDATED_ARCHES).
2. Runs the full architecture-sensitive matrix, $SPIKY_ROOT/src/spiky/lutorch/tests/test_pow2_int8.py:
   - read_cells (the cells reference) bit-exact vs pow2_read.int8_blend_read (eager and compiled) and an explicit shift-add
     sum, D = 48/40/52/8/128 x block sizes 32/64/128 x both load styles x garbage stride padding, boundary integers;
   - int32 headroom worst case;
   - read_fused (integers in-kernel) == read_cells == int8_blend_read over the same matrix;
   - drift: the op's integers == the fused kernel's integers on random margins and at the q rounding thresholds +-1 ulp;
   - the artefact's fused forward == its torch read (all block sizes, small batches); train == forward_int bit for bit;
   - recompute backward, gradcheck, skip/drop gradients, no graph break, fallback tests.
   plus test_light_quant_mode.py (the torch definition the fallback uses).
3. Writes kernel_gate.json and prints ONE verdict line:
   KERNEL GATE: PASS  -> the kernel is validated on this GPU; launch with the kernel (leave SPIKY_P2_CUDA_DISABLE unset).
   KERNEL GATE: FAIL  -> do NOT train on the kernel: set SPIKY_P2_CUDA_DISABLE=1 in run.env (torch fallback) and report
                         the failures listed here. PASS requires 0 failed, 0 errors and 0 SKIPPED kernel tests (a skip means
                         the kernel was not built / not serving, which is not a validation).
Exit code 0 on PASS, 1 on FAIL.
"""
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
spiky = os.environ.get("SPIKY_ROOT", "")
if not spiky or spiky == "FILL":
    sys.exit("validate_kernel.py: fill SPIKY_ROOT in run.env and source it first")
if os.environ.get("SPIKY_P2_CUDA_DISABLE") == "1":
    sys.exit("validate_kernel.py: SPIKY_P2_CUDA_DISABLE=1 is set; unset it for the validation run")

res = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), spiky_head=subprocess.run(["git", "-C", spiky, "rev-parse", "HEAD"],
                                                                             capture_output=True, text=True).stdout.strip())
import torch  # noqa: E402

res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
res["capability"] = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
from spiky.lutorch import pow2_int8 as K  # noqa: E402

res["validated_arches"] = [list(a) for a in K.VALIDATED_ARCHES]
t0 = time.time()
ok_build, msg = K.available()
res["build"] = dict(ok=ok_build, message=msg, seconds=round(time.time() - t0, 1))
print(f"[build] {msg} ({res['build']['seconds']} s) on {res['device']} capability {res['capability']}", flush=True)

tests = [os.path.join(spiky, "src/spiky/lutorch/tests/test_pow2_int8.py"),
         os.path.join(spiky, "src/spiky/lutorch/tests/test_light_quant_mode.py")]
xml = os.path.join(HERE, "kernel_gate_junit.xml")
env = dict(os.environ, TORCHINDUCTOR_AUTOGRAD_CACHE="0", TORCHINDUCTOR_FX_GRAPH_CACHE="0")
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", f"--junitxml={xml}", *tests],
                   cwd=spiky, env=env, capture_output=True, text=True)
tail = r.stdout.strip().splitlines()[-40:]
failed, skipped, passed = [], [], 0
for case in ET.parse(xml).getroot().iter("testcase"):
    nid = f"{os.path.basename(case.get('file') or case.get('classname', ''))}::{case.get('name')}"
    if case.find("failure") is not None or case.find("error") is not None:
        failed.append(nid)
    elif case.find("skipped") is not None:
        skipped.append(nid)
    else:
        passed += 1
res["tests"] = dict(passed=passed, failed=failed, skipped=skipped, pytest_tail=tail)
gate = ok_build and not failed and not skipped and passed > 0
res["verdict"] = "PASS" if gate else "FAIL"
json.dump(res, open(os.path.join(HERE, "kernel_gate.json"), "w"), indent=1)
print("\n".join(tail))
print(f"\n[matrix] passed {passed}, failed {len(failed)}, skipped {len(skipped)}")
for n in failed:
    print(f"  FAILED  {n}")
for n in skipped:
    print(f"  SKIPPED {n}")
if gate:
    print("\nKERNEL GATE: PASS -- the p2_int8 kernel is validated on this GPU. Launch WITH the kernel (SPIKY_P2_CUDA_DISABLE unset).")
else:
    print("\nKERNEL GATE: FAIL -- do NOT train on the kernel. Set SPIKY_P2_CUDA_DISABLE=1 in run.env (torch fallback), "
          "then preflight / launch, and REPORT the failures above (kernel_gate.json).")
sys.exit(0 if gate else 1)
