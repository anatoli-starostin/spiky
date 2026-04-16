"""
BitFlipOptimizer — Discrete optimizer for PermutationalLut binary weights.

Per-sample pair gradients → directed flip counts → scatter to weight tensor → flip.
Per-pair adaptive lr: lr / sqrt(EMA(grad_pair²))
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
        self._n_pair_slots = []
        self._proj_for_pairs = []

        # Precomputed per-module constants
        self._tpi_flat = []        # [n_tables * output_nap] pair index
        self._orient_exp = []      # [n_tables, n_entries, output_nap]
        self._tpi_exp_flat = []    # [N] pair index for every weight element
        self._table_idx = []       # [n_tables]

        for m in self.modules:
            m.enable_bitflip_cache()
            w = m.inner.projection.weights
            w.data.copy_(w.data.sign())
            w.data[w.data == 0] = 1.0

            n_heads = m.n_heads
            n_outputs = m.n_outputs
            n_pairs = n_outputs * (n_outputs - 1) // 2
            self._n_pair_slots.append(n_heads * n_pairs)

            pair_id = torch.full((n_outputs, n_outputs), -1, dtype=torch.long, device=w.device)
            pair_orient = torch.zeros((n_outputs, n_outputs), device=w.device)
            tri_i, tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1)
            for p_idx in range(n_pairs):
                ii, jj = int(tri_i[p_idx]), int(tri_j[p_idx])
                pair_id[ii, jj] = p_idx
                pair_id[jj, ii] = p_idx
                pair_orient[ii, jj] = 1.0
                pair_orient[jj, ii] = -1.0

            idx_a = m.idx_a
            idx_b = m.idx_b
            pi = pair_id[idx_a, idx_b] + \
                torch.arange(n_heads, device=w.device).unsqueeze(1) * n_pairs
            po = pair_orient[idx_a, idx_b]

            n_tables = n_heads * m.tph
            n_entries = w.shape[1]
            output_nap = m.output_nap

            tpi = pi.reshape(n_tables, output_nap)
            tpo = po.reshape(n_tables, output_nap)

            # Precompute constants
            self._tpi_flat.append(tpi.reshape(-1))
            self._orient_exp.append(tpo.unsqueeze(1).expand(n_tables, n_entries, output_nap).contiguous())
            self._tpi_exp_flat.append(
                tpi.unsqueeze(1).expand(n_tables, n_entries, output_nap).reshape(-1).contiguous()
            )
            self._table_idx.append(torch.arange(n_tables, device=w.device))

            self._pair_var.append(torch.ones(n_heads * n_pairs, device=w.device))

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
            lookup_indices = m._cached_lookup_indices
            if grad_out is None or lookup_indices is None:
                m._cached_raw = None
                m._cached_grad_output = None
                m._cached_lookup_indices = None
                continue

            n_tables = m.n_heads * m.tph
            n_entries = w.shape[1]
            output_nap = m.output_nap
            H = m.n_heads
            n_pair_slots = self._n_pair_slots[i]
            n_pairs = n_pair_slots // H
            B = grad_out.shape[0]
            dev = w.device

            tpi_flat = self._tpi_flat[i]
            orient_exp = self._orient_exp[i]
            tpi_exp_flat = self._tpi_exp_flat[i]
            table_idx = self._table_idx[i]

            # --- Steps 1-4: pair gradient → dv → split ---
            if m.return_dominance:
                grad_pair_flat = grad_out.reshape(B, n_pair_slots)
                pair_grad_sq = (grad_out ** 2).mean(dim=0).reshape(-1)
            else:
                grad_pair = torch.einsum('bhn,hnp->bhp', grad_out, self._proj_for_pairs[i])
                grad_pair_flat = grad_pair.reshape(B, n_pair_slots)
                pair_grad_sq = (grad_pair ** 2).mean(dim=0).reshape(-1)

            self._pair_var[i].mul_(beta2).add_(pair_grad_sq, alpha=1 - beta2)
            eff_lr = lr / (self._pair_var[i].sqrt() + self.eps)

            dv = -(eff_lr.unsqueeze(0) * grad_pair_flat)  # [B, H*n_pairs]
            flips_up = dv.clamp(min=0)
            flips_down = (-dv).clamp(min=0)

            # --- Step 5: pools (contribution changes each step) ---
            contribution = w.data * orient_exp  # [n_tables, n_entries, output_nap]
            contrib_flat = contribution.reshape(-1)

            pool_minus_count = torch.zeros(n_pair_slots, device=dev)
            pool_minus_count.scatter_add_(0, tpi_exp_flat, (contrib_flat < 0).float())
            pool_plus_count = torch.zeros(n_pair_slots, device=dev)
            pool_plus_count.scatter_add_(0, tpi_exp_flat, (contrib_flat > 0).float())

            # --- Step 6: per-sample prob for accessed entries ---
            # Lookup per-table pair flips: [B, n_tables, output_nap]
            sample_flips_up = flips_up[:, tpi_flat].view(B, n_tables, output_nap)
            sample_flips_down = flips_down[:, tpi_flat].view(B, n_tables, output_nap)

            # Contribution at accessed entries: [B, n_tables, output_nap]
            gathered_contrib = contribution[table_idx.unsqueeze(0), lookup_indices]

            # Pool counts per (table, output_pos): [n_tables, output_nap]
            pm = pool_minus_count[tpi_flat].view(n_tables, output_nap).clamp_(min=1)
            pp = pool_plus_count[tpi_flat].view(n_tables, output_nap).clamp_(min=1)

            # Prob: [B, n_tables, output_nap]
            prob_per_sample = (
                sample_flips_up * (gathered_contrib < 0).float() / pm.unsqueeze(0) +
                sample_flips_down * (gathered_contrib > 0).float() / pp.unsqueeze(0)
            )

            # --- Step 7: scatter to weight tensor ---
            li_exp = lookup_indices.unsqueeze(-1).expand(B, n_tables, output_nap)
            flat_target = (
                table_idx.view(1, n_tables, 1) * n_entries + li_exp
            ) * output_nap + torch.arange(output_nap, device=dev).view(1, 1, output_nap)
            # flat_target: [B, n_tables, output_nap] — broadcasts, no expand needed

            N = n_tables * n_entries * output_nap
            agg_prob = torch.zeros(N, device=dev)
            agg_prob.scatter_add_(0, flat_target.reshape(-1), prob_per_sample.reshape(-1))

            # --- Step 8: clamp and flip ---
            agg_prob.clamp_(max=1.0)
            do_flip = torch.rand(N, device=dev) < agg_prob
            w.data.view(-1)[do_flip] *= -1

            m._cached_raw = None
            m._cached_grad_output = None
            m._cached_lookup_indices = None

    def zero_grad(self):
        for m in self.modules:
            w = m.inner.projection.weights
            if w.grad is not None:
                w.grad.zero_()
