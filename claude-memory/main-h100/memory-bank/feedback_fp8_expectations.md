---
name: fp8 training expectations
description: User expected fp8 quantization might accelerate early training convergence, not just match it
type: feedback
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
When comparing fp8 vs non-fp8 experiments, user was hoping fp8 might be faster at the beginning of training (not just equivalent). The hypothesis is that fp8 quantization noise could act as regularization that helps early convergence.

**Why:** User is exploring fp8 not just for hardware efficiency but as a potential training benefit.

**How to apply:** When presenting fp8 results, note whether early training is faster/slower, not just final accuracy comparison.
