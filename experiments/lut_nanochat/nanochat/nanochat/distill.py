"""
[lut_nanochat] Online logits-distillation helpers (reusable for d24->d24 now and LUT students later).

The teacher is a frozen, eval-mode model whose full-vocab logits are the training target for a
student of (for the d24 warm-up) the same architecture. Loss = temperature-scaled KL( teacher || student ),
optionally blended with the hard-label cross-entropy.

Kept deliberately small and framework-agnostic: it only needs the two logits tensors (B, T, V) and the
targets (B, T); it does not know or care how either model was built. base_train.py wires it in behind a
`--distill-from` flag; a future LUT-student trainer can import `distillation_loss` unchanged.
"""
import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, targets, temperature=1.0, alpha=0.0):
    """
    student_logits, teacher_logits: (B, T, V) float logits (same V = full vocab, e.g. 32768).
    targets: (B, T) long; positions == -1 are padding and are ignored (matches nanochat CE ignore_index=-1).
    temperature T: softmax temperature for the soft targets (T=1 => plain softmax).
    alpha: blend with hard-label CE. loss = (1 - alpha) * KL_term + alpha * CE. alpha=0 => pure distillation.

    Returns (loss, kl_detached, ce_detached_or_None). KL is the temperature-scaled KL(teacher||student)
    (Hinton: multiplied by T^2 so its gradient magnitude is comparable across temperatures), averaged over
    non-padding tokens. The student logits carry grad; the teacher logits must already be detached (no_grad).
    """
    V = student_logits.size(-1)
    T = temperature
    s = student_logits.reshape(-1, V).float()
    t = teacher_logits.reshape(-1, V).float()
    tgt = targets.reshape(-1)
    valid = tgt != -1
    # Stable KL in LOG-SPACE for both sides. The naive F.kl_div(log_p_student, softmax(teacher)) does
    # p_t*(log p_t - log p_s) with p_t = softmax(...); at a 32768-way vocab the teacher softmax underflows
    # to exact 0.0 in many entries, and 0*log(0) = NaN. Using log_p_t (from log_softmax) and p_t = exp(log_p_t)
    # keeps every term finite: where p_t -> 0 the contribution p_t*(log_p_t - log_p_s) -> 0, no log(0).
    log_p_s = F.log_softmax(s / T, dim=-1)                       # (N, V), carries grad
    with torch.no_grad():
        log_p_t = F.log_softmax(t / T, dim=-1)                   # (N, V), teacher is frozen
        p_t = log_p_t.exp()
    per_tok_kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)         # (N,)  KL(teacher||student) per token
    n_valid = valid.sum().clamp_min(1)
    # mask padding on the reduced (N,) tensor, so the grad graph never materialises a masked (N,V) copy
    kl = torch.where(valid, per_tok_kl, torch.zeros_like(per_tok_kl)).sum() / n_valid * (T * T)
    ce_detached = None
    if alpha > 0.0:
        ce = F.cross_entropy(s, tgt, ignore_index=-1, reduction="mean")   # ignore_index handles padding, no copy
        loss = (1.0 - alpha) * kl + alpha * ce
        ce_detached = ce.detach()
    else:
        loss = kl
    return loss, kl.detach(), ce_detached


def _selftest():
    torch.manual_seed(0)
    B, T, V = 2, 8, 64
    tgt = torch.randint(0, V, (B, T))
    tgt[0, 0] = -1  # a padding position to exercise masking
    teacher = torch.randn(B, T, V)
    # identical logits -> KL ~ 0
    student = teacher.clone().requires_grad_(True)
    loss, kl, ce = distillation_loss(student, teacher, tgt, temperature=1.0, alpha=0.0)
    assert kl.item() < 1e-5, f"KL of identical logits should be ~0, got {kl.item()}"
    # different logits -> KL > 0, and gradient flows to the student
    student2 = torch.randn(B, T, V, requires_grad=True)
    loss2, kl2, ce2 = distillation_loss(student2, teacher, tgt, temperature=2.0, alpha=0.1)
    assert kl2.item() > 0.0, "KL of different logits should be > 0"
    loss2.backward()
    assert student2.grad is not None and torch.isfinite(student2.grad).all(), "student grad must be finite"
    assert ce2 is not None, "alpha>0 must return a CE term"
    # underflow regression: large vocab + large-magnitude logits => teacher softmax has exact-0 entries.
    # The naive F.kl_div(log_p_student, softmax(teacher)) would NaN here (0*log(0)); the log-space form must not.
    Vbig = 32768
    tbig = (torch.randn(4, Vbig) * 30.0)          # extreme logits -> many softmax underflows to 0.0
    sbig = (torch.randn(4, Vbig) * 30.0).requires_grad_(True)
    tgtbig = torch.randint(0, Vbig, (4,))
    loss3, kl3, _ = distillation_loss(sbig.unsqueeze(0), tbig.unsqueeze(0), tgtbig.unsqueeze(0), temperature=1.0)
    assert torch.isfinite(kl3), f"KL must be finite under softmax underflow, got {kl3.item()}"
    loss3.backward()
    assert torch.isfinite(sbig.grad).all(), "grad must be finite under softmax underflow"
    print(f"distill selftest OK: kl_identical={kl.item():.2e}, kl_diff={kl2.item():.4f}, ce={ce2.item():.4f}, "
          f"kl_underflow_finite={float(kl3):.4f}, grad_ok=True")


if __name__ == "__main__":
    _selftest()
