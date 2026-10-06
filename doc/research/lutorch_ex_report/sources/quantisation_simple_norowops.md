# quantisation_simple_norowops (source reference)

Dedicated quantisation reference shared by Anatoli (sent to Eugene). Original PDF on Google Drive:
https://drive.google.com/file/d/1fhZtgc1b4Jhdyvw1GmSPJ94Mrs7IT_TA/view
(title: quantisation_simple_norowops_45ef4360.pdf, owner anatoli.starostin@gmail.com, 2026-10-04).
Below is the extracted text (via the Drive connector) — the authoritative source for the report's
QuantisedConfidenceLUT / p2_int8 section. Note its running example uses H=4, T=128 (the pre-fork abl_47
geometry); the champion in our sweep is H=8, tph=64 — the math is geometry-agnostic.

---

# LUT read and quantisation (simplified)

**The LUT read and its quantisation.** Derives how one LUT FFN layer reads its tables and how the table
weights are stored as small integers. The table read needs only lookups, additions and bit shifts; the few
per-table scalars (score, blend) are computed in float. Each claim is *definition* (by construction) or
*verified* (checked on trained checkpoints).

**Notation.** H=4 heads, T=128 tables/head; n=8 anchor pairs, K=2^n=256 cells/table; D=48 cell/head width;
d_i = z_{a_i} − z_{b_i} anchor differences; u_i=|d_i|, u*=min_i u_i; c1 addressed cell, c2 runner-up;
W_t[c]∈R^D; s score; β,γ score params per layer; τ blend temperature per layer; k,k',q power-of-two
exponents; e_{h,c},b,o weight exponent/bits/offset; v,r blend weight / cell ratio; sg stop-gradient; E=384.

**1 What one LUT layer computes.** Token E=384 → compress (Linear) → HD numbers, H=4 heads of D=48. Each head
owns T=128 tables. Table t reads n=8 differences d_i=z_{a_i}−z_{b_i}; signs of d_i form an 8-bit address c1
into K=256 cells. Margins u_i=|d_i|; smallest u*=min u_i marks the least-certain bit; flipping it gives c2.
Every table reads BOTH cells, blends with softmax over margins:
  v = e^{u*/τ}/(e^{u*/τ}+e^{−u*/τ}) = σ(2u*/τ).            (1)
Head output: y_h = Σ_{t∈h} s_t [ v_t W_t[c1] + (1−v_t) W_t[c2] ] ∈ R^D;  out = decompress([y1;y2;y3;y4]) =
U[y1;..;y4] + bias.   (2)  NOTHING sits between the table sum and decompress: no norm, no nonlinearity, no
residual (definition). Section 5 depends on exactly this.

**2 The score.** s = (Σ_i u_i)·exp( γ Σ_i log σ(β u_i) ) = S·P^γ, S=Σ u_i, P=Π σ(β u_i)∈(0,1].  β,γ learned,
one pair per layer — exactly two params per layer. (S kept as inherited choice; a constant gain worked ~as
well.)

**3 From the blend to shifts and adds.** Round in log2; round(ξ)=⌊ξ+1/2⌋; ×2^k is a shift by k.
 Step 1 (exact): with r=(1−v)/v=e^{−2u*/τ}, s v W[c1]+(1−v)W[c2] = (sv) W[c1] + r W[c2].  (4)
 Step 2 (round r to pow2): since r=2^{−2u*/(τ ln2)}, v=1/(1+r), set
   q = clamp( ⌊ 2u*/(τ ln2) + 1/2 ⌋, 0, J ),   c_q = log2(1+2^{−q}) for q<C, else 0.   (5)
   so r≈2^{−q}, log2(1+r)≈c_q. τ frozen at deployment → 2/(τ ln2) constant. Cutoff C=8, bound J=64 (empirical,
   definitional). If q>Q=3 the second cell's weight <1/8 and it is dropped.
 Step 3 (round sv to pow2): since log2(sv)=log2 s − log2(1+r), set k' = round( log2 s − c_q ).
 Computing log2 s in the log domain (float): log2 s = log2 S + (γ/ln2) Σ_i log σ(β u_i).
 Result: one table reads 2^{k'} W[c1] + 2^{−q} W[c2] (two shifts, one add). q>3 → single-cell 2^{k'} W[c1].

**4 Training with rounded cell weights (straight-through).** Forward reads ŵ1 W[c1] + ŵ2 W[c2] with
   ŵ1 = sv·sg(2^{k'}/(sv)),   ŵ2 = s(1−v)·sg(2^{k'−q}/(s(1−v))).   (6)
 sg = stop-gradient (value kept, 0 grad). Forward values exactly 2^{k'}, 2^{k'−q} (ŵ2=0 if second cell
 dropped, both 0 if table skipped); gradients are those of sv and s(1−v) times the ratios, so β,γ and the
 margins keep learning. Train read = deployed read (verified: train-mode loss = eval-mode loss bit for bit).
 **Exponent window.** k' takes only L=8 values, k'∈[k_max−7, k_max]=[−3,4], top k_max=4 fixed → 3 bits.
 (L=8 narrowest that trained; L=6 failed, L=12 no better; k_max=4 largest measured, now fixed.) k'<−3 →
 table SKIPPED (weight 0, keeps gradient through ratios). k'>4 → CLAMPED to 4 (fires on ~0.001–0.007% of
 tables, verified).

**5 Storing weights as small integers.** Per head h and output channel c, shared power of two:
   e_{h,c} = ⌈ log2 ( max_{t∈h, cells} |W_t[·]_c| ) ⌉ − (b−1) + o.   (7)
   Ŵ = clamp( round(W/2^{e_{h,c}}), −2^{b−1}, 2^{b−1}−1 ).   (8)   Layer uses W_q = 2^{e_{h,c}} Ŵ.
 Training STE: W̃ = W_q + (W − sg W) (value W_q, grad to float W). Exponent recomputed every forward.
 int4: b=4 ([−8,7], o=−1, step halved → few largest saturate); int8: b=8 ([−128,127], o=0, never saturates).
 Offsets fixed per row, not searched.
 **Scales cost nothing at runtime.** Because e_{h,c} is the SAME for every table of head h, it leaves the
 head sum:  y_h[c] = Σ_{t∈h} ( 2^{k'_t} W_q,t[c1]_c + 2^{−q_t} W_q,t[c2]_c )
                    = 2^{e_{h,c}} Σ_{t∈h} ( 2^{k'_t} Ŵ_t[c1]_c + 2^{−q_t} Ŵ_t[c2]_c )  [integer cells only].  (9)
 Since nothing sits between y and decompress, U y = Σ_{h,c} U_{·,Dh+c} y_h[c], so 2^{e_{h,c}} folds into column
 Dh+c of U — ONCE, at export, after training, when e_{h,c} no longer changes. During training U untouched;
 scale applied inline in float via W̃. At runtime tables are pure integers; the HD·6=1152 exponents (1.2 KB)
 disappear into decompress (verified: integer read within 1e−4 bpb of float read).

**6 The whole deployed read.** ŷ_h = Σ_{t∈h} ( 2^{k'_t} Ŵ_t[c1] + 2^{−q_t} Ŵ_t[c2] );  out = U'[ŷ1;..;ŷ4]+bias,
 sum over non-skipped tables (k'_t≥−3), k'_t clamped at 4, second term dropped when q_t>3, U' = decompress with
 2^{e_{h,c}} folded into columns.
 **In integers.** ŷ_h kept in units 2^{−6}: each table adds Ŵ[c1]≪(k'+6) + Ŵ[c2]≪(k'+6−q) — both shifts ≥0
 since k'≥−3 and q≤3. The 2^{−6} folds into U' with the 2^{e_{h,c}}. At int8, |ŷ_h| ≤ 128·256·2^10 = 2^25,
 inside 32-bit (definition). Worked example: (6≪8)+(4≪7)=1536+512=2048=32·2^6.
 **Scope.** The table read uses only subtractions/sign bits (address), comparisons (window), lookups, adds,
 left shifts — no multiply. The per-table scalars (log2 s, q, c_q) are float (log + a ×γ/ln2).
 Table sizes (6 layers): 151.0 MB fp32, 37.75 MB int8 (4×), 18.87 MB int4 (8×).

**Cost (verified).** Full power-of-two read costs ~ +0.003 bpb vs the unquantised soft read at this geometry
(two seeds), an upper bound on the rounding alone (also includes the exponent window + two-cell blend).

**Verified vs definitional.** By construction: score formula; blend factorisation; q, c_q; exponent rule +
offsets; folding per-(head,channel) scales + the 2^{−6} unit into decompress; the integer shift form.
Verified on checkpoints: train-loss = eval-loss exactly; factorised read reproduces blended read's loss; clamp
rate 0.001–0.007%; integer read within 1e−4 bpb of exact; the third log-gain param moved bpb < seed noise
(so two params/layer kept).
