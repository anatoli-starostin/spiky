"""Anchor-pair sampling for lutorch_ex cartridges.

Only one policy is supported: **canonical full coverage**, a standalone port of the
gen-1 ``CANONICAL_FULL_COVERAGE`` sampler (originally
``spiky.lutorch.lut_helpers._get_canonical_full_coverage_pairs``) — reimplemented here
with zero imports from the old ``lutorch``.

For each table group the sampler draws ``nap`` *canonical* coordinate pairs ``(a, b)``
with ``a < b`` from the pool of all ``P = C(d_in, 2)`` possible pairs, using tiled
``randperm`` (sampling without replacement), so the pairs a group uses are distinct and,
when ``tph * nap >= P``, cover the whole pool. A repair pass guarantees no table ever
repeats a pair even across a tile boundary. Each group ``g`` is seeded ``seed + g`` — the
same per-head seeding the gen-1 LightMHL path uses (``random_seed + h``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _canonical_pool(d_in: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(tri_i, tri_j)`` — the canonical ``a < b`` pair pool, each ``[P]`` long."""
    tri_i, tri_j = torch.triu_indices(d_in, d_in, offset=1)
    return tri_i.to(device).long(), tri_j.to(device).long()


def _repair_intra_table_duplicates(pairs_table: torch.Tensor) -> None:
    """In-place greedy repair of intra-row (within-table) duplicate pool indices.

    Only tables straddling a ``randperm``-tile boundary can carry duplicates, so the
    affected row count is tiny in practice. For each such table, swap one duplicate slot
    with a slot in another table so both tables stay duplicate-free. Raises if no valid
    swap partner exists (pool too tight). Ported verbatim from gen-1.
    """
    n_tables, nap = pairs_table.shape
    for t in range(n_tables):
        while True:
            row_list = pairs_table[t].tolist()
            seen = {}
            dup_pos = -1
            for k, v in enumerate(row_list):
                if v in seen:
                    dup_pos = k
                    break
                seen[v] = k
            if dup_pos == -1:
                break
            dup_val = row_list[dup_pos]
            row_set_excl = set(row_list)
            row_set_excl.discard(dup_val)

            swapped = False
            for t2 in range(n_tables):
                if t2 == t:
                    continue
                row2_list = pairs_table[t2].tolist()
                if dup_val in row2_list:
                    continue
                row2_set = set(row2_list)
                for k2, v2 in enumerate(row2_list):
                    if v2 in row_set_excl:
                        continue
                    row2_set_after = (row2_set - {v2}) | {dup_val}
                    if len(row2_set_after) != len(row2_list):
                        continue
                    tmp = pairs_table[t, dup_pos].clone()
                    pairs_table[t, dup_pos] = pairs_table[t2, k2]
                    pairs_table[t2, k2] = tmp
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                raise RuntimeError(
                    f"canonical_full_coverage_pairs: failed to repair duplicate in table "
                    f"{t} pos {dup_pos} (val={dup_val}); pool too tight (n_tables={n_tables}, nap={nap})"
                )


def canonical_full_coverage_pairs(
    d_in: int,
    n_groups: int,
    tph: int,
    nap: int,
    *,
    seed: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical full-coverage anchor pairs, one independent draw per group.

    Args:
        d_in: per-head input width (anchor coordinates live in ``[0, d_in)``).
        n_groups: number of table groups (``max(h_in, h_out)`` in the lutorch_ex contract).
        tph: tables per group.
        nap: anchor pairs per table. Must satisfy ``nap <= C(d_in, 2)``.
        seed: base seed; group ``g`` is drawn with ``torch.Generator`` seeded ``seed + g``.
        device: where to build the index tensors (default CPU, for cross-device determinism).

    Returns:
        ``(anchor_a, anchor_b)``, each ``[n_groups, tph, nap]`` long, with ``a < b`` in
        every entry (canonical ordering). Within any table the ``nap`` pairs are distinct;
        within a group they cover the whole pool when ``tph * nap >= C(d_in, 2)``.
    """
    P = d_in * (d_in - 1) // 2
    if nap > P:
        raise ValueError(
            f"canonical_full_coverage_pairs: nap={nap} exceeds the number of distinct "
            f"coordinate pairs C(d_in, 2) = C({d_in}, 2) = {P}"
        )
    dev = device or torch.device("cpu")
    tri_i, tri_j = _canonical_pool(d_in, dev)

    a = torch.empty(n_groups, tph, nap, dtype=torch.long, device=dev)
    b = torch.empty(n_groups, tph, nap, dtype=torch.long, device=dev)
    slots = tph * nap
    repeats = (slots + P - 1) // P
    for g in range(n_groups):
        gen = torch.Generator(device=dev).manual_seed(int(seed) + g)
        perm = torch.cat(
            [torch.randperm(P, device=dev, generator=gen) for _ in range(repeats)]
        )[:slots]
        pairs = perm.view(tph, nap).contiguous()
        _repair_intra_table_duplicates(pairs)
        a[g] = tri_i[pairs]
        b[g] = tri_j[pairs]
    return a, b
