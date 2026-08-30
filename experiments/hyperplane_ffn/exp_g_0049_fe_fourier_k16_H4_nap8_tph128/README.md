# exp_g_0049 — function-emitting cells, 16 Fourier harmonics per cell (STAGED, NEVER RUN)

Built, smoked and measured; never launched. A cell stores `2N+1 = 33` amplitudes over the
output-index axis instead of 384 values — **11.64x fewer** — and its output is

    W[c, i] = dc[c] + sum_n a_cos[c,n] cos(2*pi*n*i/D) + a_sin[c,n] sin(2*pi*n*i/D)

54,667,020 params against the raw fork's 330,704,652. Measured 1.170 s/step and 11.76 GiB
on gpustar, against exp_n_0136's own 1.131 s/step and 14.73 GiB on the same box — so it is
1.04x the baseline's time at *less* memory.

## Why it was not run

With **fixed** frequencies the basis matrix is shared by every cell, so

    sum_c (A[c] @ Phi) = (sum_c A[c]) @ Phi

— the synthesis **commutes with the gather-sum**. Verified numerically, not argued: relative
difference 2.40e-07 for fixed frequencies, against 1.51 for learned per-cell frequencies and
1.76 for the gaussian basis.

That makes it exactly a CompressionMHL decompress whose Linear is frozen to a DCT/DFT basis.
A learned Linear can represent any fixed basis, so on expressiveness this is a **strict
subset** of exp_n_0121 and of the exp_n_0138 decompress-only ablation. It is a compactness
result, not a quality contender. `fe_learn_freq=true` breaks the factorisation at ~2.7x the
step cost if the per-cell version is ever wanted.
