"""Anchor sampling for lutorch_ex cartridges.

**Canonical full coverage** is the one policy, provided in two flavours that mirror each
other exactly — both standalone ports (zero imports from the old ``lutorch``):

* :func:`canonical_full_coverage_pairs` — the gen-1 ``CANONICAL_FULL_COVERAGE`` sampler
  (originally ``spiky.lutorch.lut_helpers._get_canonical_full_coverage_pairs``). For each
  table group it draws ``nap`` *canonical* coordinate pairs ``(a, b)`` with ``a < b`` from
  the pool of all ``P = C(d_in, 2)`` possible pairs.

* :func:`canonical_full_coverage_singles` — the single-anchor counterpart used by
  ``anchor_mode == "single"``. For each table group it draws ``nap`` *single* coordinates
  ``a`` from the pool of all ``d_in`` coordinates (the pool is ``range(d_in)`` instead of
  the ``C(d_in, 2)`` pairs). Everything else — tiled ``randperm`` sampling without
  replacement, the intra-table duplicate repair, and the per-group seeding — is identical,
  so a group's single anchors are distinct per table and, when ``tph * nap >= d_in``,
  cover the whole coordinate pool.

Both use tiled ``randperm`` (sampling without replacement) so the elements a group uses
are distinct and, when ``tph * nap`` reaches the pool size, cover the whole pool. A repair
pass guarantees no table ever repeats an element even across a tile boundary. Each group
``g`` is seeded ``seed + g`` — the same per-head seeding the gen-1 LightMHL path uses
(``random_seed + h``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _canonical_pool(d_in: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(tri_i, tri_j)`` — the canonical ``a < b`` pair pool, each ``[P]`` long.

    Built directly on ``device`` (not created on the default device then ``.to(device)``) so the
    construction is meta-device-safe — a meta-device skeleton build (deployment load) would otherwise
    hit "cannot copy out of meta tensor" on the ``.to``. Identical result on every real device."""
    idx = torch.triu_indices(d_in, d_in, offset=1, device=device).long()
    return idx[0], idx[1]


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


def _sample_pool_indices(
    pool_size: int,
    n_groups: int,
    tph: int,
    nap: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw per-table distinct pool indices, one independent draw per group.

    Returns ``[n_groups, tph, nap]`` long indices into a pool of ``pool_size`` elements.
    Within any table the ``nap`` indices are distinct (tiled ``randperm`` + the intra-table
    repair); within a group they cover the whole pool when ``tph * nap >= pool_size``. Each
    group ``g`` is seeded ``seed + g``. Shared by both anchor flavours (pairs: pool =
    ``C(d_in, 2)``; singles: pool = ``d_in``) so they sample identically.
    """
    out = torch.empty(n_groups, tph, nap, dtype=torch.long, device=device)
    slots = tph * nap
    repeats = (slots + pool_size - 1) // pool_size
    for g in range(n_groups):
        gen = torch.Generator(device=device).manual_seed(int(seed) + g)
        perm = torch.cat(
            [torch.randperm(pool_size, device=device, generator=gen) for _ in range(repeats)]
        )[:slots]
        pool_idx = perm.view(tph, nap).contiguous()
        _repair_intra_table_duplicates(pool_idx)
        out[g] = pool_idx
    return out


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
    pairs = _sample_pool_indices(P, n_groups, tph, nap, seed=seed, device=dev)
    return tri_i[pairs], tri_j[pairs]


def canonical_full_coverage_singles(
    d_in: int,
    n_groups: int,
    tph: int,
    nap: int,
    *,
    seed: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Canonical full-coverage *single* anchors, one independent draw per group.

    The single-anchor counterpart of :func:`canonical_full_coverage_pairs`: each address
    bit tests one coordinate against zero (``[z[a] > eps]``) rather than a pair, so the
    pool is the ``d_in`` coordinates themselves instead of the ``C(d_in, 2)`` pairs.

    Args:
        d_in: per-head input width (anchor coordinates live in ``[0, d_in)``).
        n_groups: number of table groups (``max(h_in, h_out)`` in the lutorch_ex contract).
        tph: tables per group.
        nap: single anchors per table. Must satisfy ``nap <= d_in``.
        seed: base seed; group ``g`` is drawn with ``torch.Generator`` seeded ``seed + g``.
        device: where to build the index tensor (default CPU, for cross-device determinism).

    Returns:
        ``anchor_a``, ``[n_groups, tph, nap]`` long, with every entry in ``[0, d_in)``.
        Within any table the ``nap`` coordinates are distinct; within a group they cover
        every coordinate when ``tph * nap >= d_in`` (a permutation of ``range(d_in)`` per
        group when ``tph * nap == d_in``).
    """
    if nap > d_in:
        raise ValueError(
            f"canonical_full_coverage_singles: nap={nap} exceeds the number of distinct "
            f"coordinates d_in={d_in}"
        )
    dev = device or torch.device("cpu")
    return _sample_pool_indices(d_in, n_groups, tph, nap, seed=seed, device=dev)
