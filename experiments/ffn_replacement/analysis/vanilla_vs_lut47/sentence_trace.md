## Single-sentence trace: vanilla (dense MLP FFN) vs LUT47 (LUT FFN)


### Sentence 1: `The capital of France is`
Tokens: ['The', ' capital', ' of', ' France', ' is']


**Next-token prediction after the full prompt (top-5):**

- vanilla: ' of'=0.16, ' is'=0.05, ' in'=0.04, ' not'=0.04, ' a'=0.04
- LUT47: ' a'=0.12, ' the'=0.11, ' an'=0.02, ' of'=0.02, ' '=0.02

**Per-position top-1 next-token (each model), showing agree/disagree:**

  [0] after 'The': vanilla->'The'  LUT->'The'  =
  [1] after ' capital': vanilla->' is'  LUT->'The'  X
  [2] after ' of': vanilla->' the'  LUT->' the'  =
  [3] after ' France': vanilla->' is'  LUT->' France'  X
  [4] after ' is': vanilla->' of'  LUT->' a'  X

**Per-layer norms at the FINAL position (attn delta / FFN delta / residual):**

- vanilla:
    L0: attn|d|=0.28  ffn|d|=4.61  resid=4.94  (ffn/resid=0.93)
    L1: attn|d|=3.39  ffn|d|=2.70  resid=6.36  (ffn/resid=0.42)
    L2: attn|d|=4.48  ffn|d|=2.81  resid=7.77  (ffn/resid=0.36)
    L3: attn|d|=6.12  ffn|d|=4.90  resid=9.45  (ffn/resid=0.52)
    L4: attn|d|=7.18  ffn|d|=5.74  resid=11.35  (ffn/resid=0.51)
    L5: attn|d|=11.40  ffn|d|=10.38  resid=19.56  (ffn/resid=0.53)
- LUT47:
    L0: attn|d|=0.30  ffn|d|=0.42  resid=0.91  (ffn/resid=0.46)
    L1: attn|d|=2.38  ffn|d|=1.14  resid=3.31  (ffn/resid=0.34)
    L2: attn|d|=2.05  ffn|d|=1.05  resid=4.39  (ffn/resid=0.24)
    L3: attn|d|=2.47  ffn|d|=1.09  resid=5.03  (ffn/resid=0.22)
    L4: attn|d|=3.83  ffn|d|=2.08  resid=6.64  (ffn/resid=0.31)
    L5: attn|d|=9.53  ffn|d|=2.38  resid=12.22  (ffn/resid=0.19)

**Layer-5 attention from the final token (top-3 attended tokens per head):**

- vanilla:
    head0: ' capital'(0.39), ' of'(0.29), ' is'(0.19)
    head1: ' is'(0.37), ' capital'(0.35), ' France'(0.14)
    head2: ' is'(0.82), ' of'(0.08), ' France'(0.05)
    head3: ' is'(0.51), ' of'(0.47), 'The'(0.01)
    head4: ' of'(0.43), ' is'(0.26), 'The'(0.20)
    head5: ' is'(0.66), ' of'(0.15), 'The'(0.11)
- LUT47:
    head0: ' is'(0.90), ' France'(0.07), ' capital'(0.01)
    head1: ' is'(0.63), ' France'(0.14), ' capital'(0.09)
    head2: ' is'(0.78), ' France'(0.13), ' of'(0.05)
    head3: ' is'(0.91), ' France'(0.05), ' of'(0.02)
    head4: ' is'(0.65), ' of'(0.21), ' France'(0.12)
    head5: ' is'(0.77), ' France'(0.12), ' capital'(0.05)

### Sentence 2: `The cat sat on the mat.`
Tokens: ['The', ' cat', ' sat', ' on', ' the', ' mat', '.']


**Next-token prediction after the full prompt (top-5):**

- vanilla: '.'=0.38, '.

'=0.08, '<|bos|>'=0.04, ' and'=0.03, ','=0.02
- LUT47: '.'=0.06, '<|bos|>'=0.04, ' The'=0.01, ' .'=0.01, ' and'=0.01

**Per-position top-1 next-token (each model), showing agree/disagree:**

  [0] after 'The': vanilla->'The'  LUT->'The'  =
  [1] after ' cat': vanilla->'

'  LUT->' cat'  X
  [2] after ' sat': vanilla->' sat'  LUT->' sat'  =
  [3] after ' on': vanilla->' the'  LUT->' on'  X
  [4] after ' the': vanilla->' the'  LUT->' side'  X
  [5] after ' mat': vanilla->' on'  LUT->' on'  =
  [6] after '.': vanilla->'.'  LUT->'.'  =

**Per-layer norms at the FINAL position (attn delta / FFN delta / residual):**

- vanilla:
    L0: attn|d|=0.28  ffn|d|=4.56  resid=4.73  (ffn/resid=0.97)
    L1: attn|d|=3.05  ffn|d|=2.38  resid=5.78  (ffn/resid=0.41)
    L2: attn|d|=4.17  ffn|d|=3.15  resid=7.12  (ffn/resid=0.44)
    L3: attn|d|=7.97  ffn|d|=5.15  resid=8.59  (ffn/resid=0.60)
    L4: attn|d|=8.59  ffn|d|=8.33  resid=10.89  (ffn/resid=0.76)
    L5: attn|d|=11.34  ffn|d|=9.09  resid=16.40  (ffn/resid=0.55)
- LUT47:
    L0: attn|d|=0.36  ffn|d|=0.86  resid=1.25  (ffn/resid=0.69)
    L1: attn|d|=1.84  ffn|d|=1.01  resid=2.82  (ffn/resid=0.36)
    L2: attn|d|=1.84  ffn|d|=1.01  resid=3.54  (ffn/resid=0.28)
    L3: attn|d|=2.51  ffn|d|=1.14  resid=4.48  (ffn/resid=0.26)
    L4: attn|d|=4.07  ffn|d|=1.50  resid=5.67  (ffn/resid=0.26)
    L5: attn|d|=9.25  ffn|d|=2.05  resid=11.17  (ffn/resid=0.18)

**Layer-5 attention from the final token (top-3 attended tokens per head):**

- vanilla:
    head0: ' sat'(0.69), ' on'(0.08), ' mat'(0.08)
    head1: '.'(0.37), 'The'(0.27), ' mat'(0.10)
    head2: '.'(0.53), ' sat'(0.30), ' on'(0.07)
    head3: '.'(0.71), ' sat'(0.18), ' on'(0.03)
    head4: ' sat'(0.37), '.'(0.25), 'The'(0.23)
    head5: ' sat'(0.46), '.'(0.37), 'The'(0.09)
- LUT47:
    head0: '.'(0.84), ' mat'(0.05), ' the'(0.04)
    head1: '.'(0.47), ' sat'(0.21), ' on'(0.10)
    head2: '.'(0.65), ' mat'(0.15), ' sat'(0.08)
    head3: ' mat'(0.32), '.'(0.30), ' the'(0.18)
    head4: '.'(0.88), ' mat'(0.05), ' the'(0.03)
    head5: '.'(0.68), ' mat'(0.17), ' the'(0.05)