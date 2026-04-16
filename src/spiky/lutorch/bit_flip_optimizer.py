"""
BitFlipOptimizer — Discrete optimizer for PermutationalLut binary weights.

Per-sample pair gradients → directed flip counts → batch reduce → execute.
Uses weight gradient (grad*w > 0) for flip direction (verified correct),
with per-pair adaptive lr from output gradient variance.

Usage:
    perm_modules = [layer.q_perm, layer.k_perm, ...]
    bit_opt = BitFlipOptimizer(perm_modules, lr=0.001)

    loss.backward()
    adam_optimizer.step()
    bit_opt.step()
    adam_optimizer.zero_grad()
"""
import torch


class BitFlipOptimizer:
    def __init__(self, perm_modules, lr=0.001, lr_schedule_fn=None,
                 beta2=0.999, eps=1e-8):
        self.modules = list(perm_modules)
        self.lr = lr
        self.lr_schedule_fn = lr_schedule_fn
        self.beta2 = beta2
        self.eps = eps
        self._step_count = 0

        self._pair_var = []
        self._pair_maps = []
        self._n_pair_slots = []
        self._proj_for_pairs = []

        for m in self.modules:
            m.enable_bitflip_cache()
            w = m.inner.projection.weights
            w.data.copy_(w.data.sign())
            w.data[w.data == 0] = 1.0

            n_heads = m.n_heads
            n_outputs = m.n_outputs
            n_pairs = n_outputs * (n_outputs - 1) // 2
            self._n_pair_slots.append(n_heads * n_pairs)

            # Build pair map for adaptive variance
            pair_id = torch.full((n_outputs, n_outputs), -1, dtype=torch.long, device=w.device)
            tri_i, tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1)
            for p_idx in range(n_pairs):
                ii, jj = int(tri_i[p_idx]), int(tri_j[p_idx])
                pair_id[ii, jj] = p_idx
                pair_id[jj, ii] = p_idx

            idx_a = m.idx_a
            idx_b = m.idx_b
            pair_indices = pair_id[idx_a, idx_b] + \
                torch.arange(n_heads, device=w.device).unsqueeze(1) * n_pairs

            n_tables = n_heads * m.tph
            n_entries = w.shape[1]
            pair_map = pair_indices.reshape(n_tables, m.output_nap) \
                .unsqueeze(1).expand(n_tables, n_entries, m.output_nap).reshape(-1)
            self._pair_maps.append(pair_map)
            self._pair_var.append(torch.ones(n_heads * n_pairs, device=w.device))

            # Projection for output → pair gradient
            if m.return_dominance:
                self._proj_for_pairs.append(None)
            else:
                proj_pair = torch.zeros(n_heads, n_outputs, n_pairs, device=w.device)
                for p_idx in range(n_pairs):
                    ii, jj = int(tri_i[p_idx]), int(tri_j[p_idx])
                    proj_pair[:, ii, p_idx] = 1.0
                    proj_pair[:, jj, p_idx] = -1.0
                self._proj_for_pairs.append(proj_pair)

    @torch.no_grad()
    def step(self):
        self._step_count += 1
        lr = self.lr
        if self.lr_schedule_fn is not None:
            lr = self.lr * self.lr_schedule_fn(self._step_count)
        beta2 = self.beta2

        for i, m in enumerate(self.modules):
            w = m.inner.projection.weights
            grad_out = m._cached_grad_output
            if w.grad is None:
                m._cached_raw = None
                m._cached_grad_output = None
                continue

            flat_w = w.data.view(-1)
            flat_g = w.grad.view(-1)
            pair_map = self._pair_maps[i]
            N = flat_w.numel()

            # Flip direction from weight gradient (verified: grad*w > 0 = correct)
            flip_pressure = flat_g * flat_w
            want_flip = flip_pressure > 0
            flip_score = flat_g.abs() * want_flip.float()

            # Per-pair adaptive lr from output gradient
            if grad_out is not None:
                H = m.n_heads
                n_pairs = self._n_pair_slots[i] // H

                if m.return_dominance:
                    grad_pair = grad_out  # [B, H, n_pairs]
                else:
                    grad_pair = torch.einsum('bhn,hnp->bhp', grad_out, self._proj_for_pairs[i])

                # Update per-pair variance
                pair_grad_sq = (grad_pair ** 2).mean(dim=0).reshape(-1)
                self._pair_var[i].mul_(beta2).add_(pair_grad_sq, alpha=1 - beta2)

                # Per-entry adaptive scale from pair variance
                pair_std = self._pair_var[i].sqrt() + self.eps
                entry_scale = 1.0 / pair_std[pair_map]  # [N]
                flip_score = flip_score * entry_scale

            # Budget: lr * total_votes
            budget = lr * m.n_heads * m.tph * m.output_nap

            # Normalize flip_score to probability summing to budget
            score_sum = flip_score.sum()
            if score_sum > 0:
                flip_prob = (flip_score * (budget / score_sum)).clamp(max=1.0)
                do_flip = torch.rand(N, device=w.device) < flip_prob
                flat_w[do_flip] *= -1

            m._cached_raw = None
            m._cached_grad_output = None

    def zero_grad(self):
        for m in self.modules:
            w = m.inner.projection.weights
            if w.grad is not None:
                w.grad.zero_()
