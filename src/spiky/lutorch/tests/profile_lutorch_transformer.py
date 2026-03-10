"""
Profiling script for LUTorch-based transformer.

Replicates the training job from `workbooks/lutorch_transformer.ipynb` in a
distilled form and runs a grid of configurations:

- smooth_mode: True / False
- n_alternatives: 1 / 3 / "all" (== n_anchor_pairs)
- Backends (GPU only when CUDA available): gpu_pure, gpu_compiled_no_k,
  gpu_compiled_anchor, gpu_compiled_all (see BackendConfig).

For each configuration the script:
  - loads text snippets via TextSnippetSampler, builds LUTTransformer
  - runs warmup steps then a timed loop: forward, backward, optimizer step
  - reports mean wall-clock time per step (ms); on CUDA, torch.cuda.synchronize()
    is used so times reflect GPU work
  - when custom kernels are used, fetches native LUTorchManager profiling stats

Results are written to `lutorch_profile_results.md` (summary table + per-run
details with native stats when present). Run from repo root with PYTHONPATH=src:

    python -m spiky.lutorch.tests.profile_lutorch_transformer
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from spiky.util.text_snippet_sampler import TextSnippetSampler

# Import LUTorch modules via module aliases so that importlib.reload can
# re-evaluate environment-driven flags (_USE_LUTORCH_COMPILE, custom CUDA, etc.).
import importlib

import spiky.lutorch.anchor_pairs_lookup as anchor_pairs_lookup_mod
import spiky.lutorch.l_projection as l_projection_mod
import spiky.lutorch.multi_head_lut as mhl
import spiky.lutorch.lut_cross_attention as lca


# --- Constants and simple caches matching lutorch_transformer.ipynb ---

CONTEXT_SIZE = 32
RAW_VOCAB_SIZE = 256
BOS_ID = RAW_VOCAB_SIZE
VOCAB_SIZE = RAW_VOCAB_SIZE + 1

# Cache TextSnippetSampler per device so we reuse the same sampling regions
# across all grid runs on that device.
_SNIPPET_SAMPLER_CACHE: Dict[torch.device, TextSnippetSampler] = {}


@dataclass(frozen=True)
class LUTTransformerConfig:
    vocab_size: int = VOCAB_SIZE
    embedding_dim: int = 32
    num_layers: int = 6
    num_heads: int = 4
    n_anchor_pairs_attn: int = 14
    n_anchor_pairs_ffn: int = 14
    n_positional_buckets: int = 8
    tables_per_head_attn: int = 32
    tables_per_head_value: int = 16
    ffn_tables: int = 16
    dropout: float = 0.0
    smooth_mode: bool = True
    n_alternatives: int = 1
    device: object = torch.device("cpu")
    connected_anchors_mode: bool = False
    random_seed: int = 42
    attention_temperature: float = 0.25
    embedding_temperature: float = 0.1
    initial_weights_noise: float = 0.001
    uncertainty_mode: "mhl.UncertaintyMode" = None  # type: ignore[assignment]
    pair_config: "lca.PairProcessingConfig" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.uncertainty_mode is None:
            object.__setattr__(
                self,
                "uncertainty_mode",
                mhl.UncertaintyMode.INVERSE_L1,  # type: ignore[attr-defined]
            )
        if self.pair_config is None:
            object.__setattr__(self, "pair_config", lca.PairProcessingConfig())
        assert (self.embedding_dim % self.num_heads) == 0


class LUTTransformer(nn.Module):
    """Transformer with LUTorch primitives: MultiHeadLut + LUTCrossAttention."""

    class Block(nn.Module):
        def __init__(self, c: LUTTransformerConfig) -> None:
            super().__init__()
            self.cross_attn = lca.LUTCrossAttention(
                mhl.MultiHeadLut(
                    input_dim=c.embedding_dim,
                    n_heads=c.num_heads,
                    n_outputs=1,
                    n_anchor_pairs=c.n_anchor_pairs_attn,
                    tables_per_head=c.tables_per_head_attn,
                    n_buckets=c.n_positional_buckets,
                    smooth_mode=c.smooth_mode,
                    n_alternatives=c.n_alternatives,
                    device=c.device,
                    connected_anchors_mode=c.connected_anchors_mode,
                    random_seed=c.random_seed,
                    initial_weights_noise=c.initial_weights_noise,
                    uncertainty_mode=c.uncertainty_mode,
                ),
                causal=True,
                attention_temperature=c.attention_temperature,
                n_positional_buckets=c.n_positional_buckets,
                pair_config=c.pair_config,
            )
            self.value_lut = mhl.MultiHeadLut(
                input_dim=c.embedding_dim,
                n_heads=c.num_heads,
                n_outputs=c.embedding_dim // c.num_heads,
                n_anchor_pairs=c.n_anchor_pairs_attn,
                tables_per_head=c.tables_per_head_value,
                smooth_mode=c.smooth_mode,
                n_alternatives=c.n_alternatives,
                device=c.device,
                connected_anchors_mode=c.connected_anchors_mode,
                random_seed=c.random_seed,
                initial_weights_noise=c.initial_weights_noise,
                uncertainty_mode=c.uncertainty_mode,
            )
            self.attn_dropout = nn.Dropout(c.dropout)
            self.ffn = mhl.MultiHeadLut(
                input_dim=c.embedding_dim,
                n_heads=1,
                n_outputs=c.embedding_dim,
                n_anchor_pairs=c.n_anchor_pairs_ffn,
                tables_per_head=c.ffn_tables,
                smooth_mode=c.smooth_mode,
                n_alternatives=c.n_alternatives,
                device=c.device,
                connected_anchors_mode=c.connected_anchors_mode,
                random_seed=c.random_seed,
                initial_weights_noise=c.initial_weights_noise,
                uncertainty_mode=c.uncertainty_mode,
            )
            self.ffn_dropout = nn.Dropout(c.dropout)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            B, S, E = z.shape
            attn_weights = self.cross_attn(z, z)  # [B, S, S, H]
            v = self.value_lut(z.reshape(-1, E))  # [B*S, H, E//H]
            H = v.shape[1]
            v = v.reshape(B, S, H, -1)  # [B, S, H, E//H]
            attn_out = attn_weights.permute(0, 3, 1, 2) @ v.permute(
                0, 2, 1, 3
            )  # [B, H, S, E//H]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, E)  # [B, S, E]
            z = z + self.attn_dropout(attn_out)
            ffn_out = self.ffn(z.reshape(-1, E)).reshape(B, S, -1)
            z = z + self.ffn_dropout(ffn_out)
            return z

    def __init__(self, c: LUTTransformerConfig) -> None:
        super().__init__()
        self.config = c
        dev = torch.device(c.device)
        with torch.no_grad():
            self.token_embedder = nn.Embedding(
                c.vocab_size, c.embedding_dim, device=dev
            )
            self.token_embedder.weight.copy_(
                torch.randn(
                    self.token_embedder.weight.shape, device=dev
                )
                * 0.1
            )
        self.layers = nn.ModuleList(
            [LUTTransformer.Block(c) for _ in range(c.num_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.token_embedder(tokens)  # [B, S, E]
        for layer in self.layers:
            z = layer(z)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        logits = (
            z @ self.token_embedder.weight.T / self.config.embedding_temperature
        )
        return logits


@dataclass
class BackendConfig:
    name: str
    device: str  # "cpu" or "cuda"
    use_torch_compile: bool
    custom_kernel_mode: str  # "none", "anchor_only", "all"


def _configure_lutorch_environment(backend: BackendConfig) -> None:
    """Configure env variables and reload LUTorch modules for a backend config."""
    # Torch.compile toggles for LUTorch internals.
    # Compilation is only meaningful (and stable) on GPU; force it OFF on CPU.
    if backend.device == "cuda" and backend.use_torch_compile:
        os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "0"
    else:
        os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"

    # Custom CUDA kernels:
    # - global switch affects anchor lookup + LProjection
    # - LProjection has an extra per-module switch
    if backend.custom_kernel_mode == "none":
        os.environ["SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS"] = "1"
        os.environ["SPIKY_LUTORCH_NO_LPROJECTION_CUSTOM_CUDA_KERNELS"] = "1"
    elif backend.custom_kernel_mode == "anchor_only":
        os.environ["SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS"] = "0"
        os.environ["SPIKY_LUTORCH_NO_LPROJECTION_CUSTOM_CUDA_KERNELS"] = "1"
    elif backend.custom_kernel_mode == "all":
        os.environ["SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS"] = "0"
        os.environ["SPIKY_LUTORCH_NO_LPROJECTION_CUSTOM_CUDA_KERNELS"] = "0"
    else:
        raise ValueError(f"Unknown custom_kernel_mode: {backend.custom_kernel_mode}")

    # Reload modules so they re-read env-controlled flags. Must reload
    # anchor_pairs_lookup and l_projection first (they read NO_CUSTOM_CUDA_KERNELS
    # at import time); then mhl/lca so they pick up the updated submodules.
    global mhl, lca
    importlib.reload(anchor_pairs_lookup_mod)
    importlib.reload(l_projection_mod)
    mhl = importlib.reload(mhl)
    lca = importlib.reload(lca)


def _build_snippet_sampler(
    device: torch.device,
    text_path: str = "fineweb_texts.txt",
    n_test_regions: int = 10_000,
    random_seed: int = 1,
) -> TextSnippetSampler:
    """
    Create TextSnippetSampler, downloading the text file if needed.

    This keeps the profiling job faithful to the notebook workload. If the
    text file is missing, we reuse the download logic from
    workbooks/lutorch_transformer.ipynb (via gdown) and then construct the
    sampler. If download fails, a FileNotFoundError is raised.

    Samplers are cached per-device so that all runs on the same device share
    the same testing regions.
    """
    # Per-device cache: avoids reloading text and re-sampling regions on every run.
    cached = _SNIPPET_SAMPLER_CACHE.get(device)
    if cached is not None:
        return cached

    if not os.path.exists(text_path):
        try:
            import gdown  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - environment-specific
            raise FileNotFoundError(
                f"Expected text file '{text_path}' for profiling workload and "
                "could not import gdown to download it. "
                "See workbooks/lutorch_transformer.ipynb for setup."
            ) from exc

        url = (
            "https://drive.google.com/file/d/"
            "1vWjyIpU6wvCPtx2OdV_4M3MXX6FsOpxV/view"
        )
        print(f"[INFO] Downloading '{text_path}' via gdown...")
        gdown.download(url, text_path, quiet=False, fuzzy=True)
        size = os.path.getsize(text_path)
        if size < 233 * 1024 * 1024:
            raise FileNotFoundError(
                f"Download of '{text_path}' failed: size too small "
                f"({size/1024/1024:.1f} MB). "
                "See workbooks/lutorch_transformer.ipynb for manual setup."
            )

    sampler = TextSnippetSampler(
        text_file_name=text_path,
        context_size=CONTEXT_SIZE,
        n_test_regions=n_test_regions,
        device=device,
        random_seed=random_seed,
    )
    _SNIPPET_SAMPLER_CACHE[device] = sampler
    return sampler


def _sample_training_batch(
    batch_size: int,
    device: torch.device,
    sampler: TextSnippetSampler,
) -> torch.Tensor:
    x = sampler.sample_training_batch(batch_size)
    x = x.to(device=device, dtype=torch.long)
    return x


def _evaluate_untrained(
    model: nn.Module,
    sampler: TextSnippetSampler,
    device: torch.device,
    batch_size: int = 128,
    last_position_only: bool = True,
    max_batches: int = 16,
) -> float:
    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for i, batch in enumerate(sampler.testing_batches_iterator(batch_size)):
            batch = batch.to(device=device, dtype=torch.long)
            inp = torch.empty_like(batch)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1]
            tgt = batch
            logits = model(inp)
            B, T, V = logits.shape
            if last_position_only:
                loss = F.cross_entropy(
                    logits[:, -1, :], tgt[:, -1], reduction="mean"
                )
                losses.append(loss.item())
            else:
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    tgt.reshape(B * T),
                    reduction="none",
                ).sum()
                losses.append(loss.item() / (CONTEXT_SIZE * B))
            if i + 1 >= max_batches:
                break

    model.train()
    return float(sum(losses) / max(len(losses), 1))


def _get_native_lutorch_stats() -> Optional[str]:
    """Fetch native LUTorchManager profiling stats if available and enabled."""
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]

        mgr = get_lutorch_manager()
        if hasattr(mgr, "get_profiling_stats"):
            return str(mgr.get_profiling_stats())
        return None
    except Exception:
        return None


def run_single_configuration(
    backend: BackendConfig,
    smooth_mode: bool,
    n_alternatives_spec: str,
    batch_size: int = 128,
    warmup_steps: int = 10,
    profile_steps: int = 100,
) -> Dict[str, object]:
    """Run one configuration and return raw profiling data."""
    print(
        f"[CONFIG] backend={backend.name}, device={backend.device}, "
        f"smooth_mode={smooth_mode}, n_alternatives_spec={n_alternatives_spec}, "
        f"batch_size={batch_size}, warmup_steps={warmup_steps}, profile_steps={profile_steps}",
        flush=True,
    )

    torch.manual_seed(1)

    device = (
        torch.device("cuda")
        if backend.device == "cuda"
        else torch.device("cpu")
    )

    # Configure LUTorch env for this backend and reload modules.
    print("[DEBUG] configuring LUTorch environment...", flush=True)
    _configure_lutorch_environment(backend)

    # Use fewer testing regions on CPU so sampler construction is faster.
    print("[DEBUG] building TextSnippetSampler...", flush=True)
    sampler = _build_snippet_sampler(
        device,
        n_test_regions=200 if backend.device == "cpu" else 10_000,
    )
    print("[DEBUG] sampler ready", flush=True)

    # Determine n_alternatives, respecting the number of anchor pairs for
    # the chosen backend/model size.
    if backend.device == "cpu":
        max_anchor_pairs = 8
    else:
        max_anchor_pairs = 14

    if n_alternatives_spec == "all":
        n_alternatives = max_anchor_pairs
    else:
        n_alternatives = int(n_alternatives_spec)
        if n_alternatives > max_anchor_pairs:
            raise ValueError(
                f"n_alternatives ({n_alternatives}) must be <= max_anchor_pairs "
                f"({max_anchor_pairs}) for backend {backend.name}"
            )

    # Use a much smaller transformer on CPU so profiling remains tractable,
    # while keeping the larger notebook-style config on GPU. Torch compilation
    # for LUTorch internals is controlled exclusively via the
    # SPIKY_LUTORCH_NO_COMPILE env var set in _configure_lutorch_environment.
    if backend.device == "cpu":
        cfg = LUTTransformerConfig(
            num_layers=2,
            num_heads=2,
            n_anchor_pairs_attn=max_anchor_pairs,
            n_anchor_pairs_ffn=max_anchor_pairs,
            n_positional_buckets=4,
            tables_per_head_attn=8,
            tables_per_head_value=8,
            ffn_tables=8,
            smooth_mode=smooth_mode,
            n_alternatives=n_alternatives,
            device=device,
        )
    else:
        cfg = LUTTransformerConfig(
            smooth_mode=smooth_mode,
            n_alternatives=n_alternatives,
            device=device,
        )
    print("[DEBUG] constructing LUTTransformer...", flush=True)
    model = LUTTransformer(cfg).to(device=device)
    print("[DEBUG] model constructed", flush=True)

    optimizer_layers = torch.optim.SGD(
        model.layers.parameters(), lr=1.0
    )
    optimizer_embedder = torch.optim.Adam(
        model.token_embedder.parameters(), lr=0.01
    )

    # Evaluate untrained; keep eval lightweight on CPU.
    print("[DEBUG] starting untrained evaluation...", flush=True)
    untrained_loss = _evaluate_untrained(
        model,
        sampler,
        device,
        batch_size=batch_size,
        last_position_only=True,
        max_batches=2 if backend.device == "cpu" else 8,
    )
    print(f"[DEBUG] untrained evaluation done, loss={untrained_loss:.6f}", flush=True)

    # Warmup iterations (no profiling).
    print("[DEBUG] starting warmup iterations (no profiling)...", flush=True)
    for _ in range(warmup_steps):
        x = _sample_training_batch(batch_size, device, sampler)
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            tgt.reshape(B * T),
            reduction="sum",
        )

        optimizer_layers.zero_grad(set_to_none=True)
        optimizer_embedder.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_layers.step()
        optimizer_embedder.step()
    print("[DEBUG] warmup iterations finished", flush=True)

    # Timed loop: forward, backward, optimizer step (sync on CUDA for accurate GPU time).
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    print("[DEBUG] starting timed loop...", flush=True)
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        mgr = get_lutorch_manager()
        mgr.reset_profiling_stats()
    except Exception:
        pass

    forward_ms_list: List[float] = []
    backward_ms_list: List[float] = []
    optimizer_step_ms_list: List[float] = []

    for step in tqdm(
        range(profile_steps),
        desc=f"{backend.name}, smooth={smooth_mode}, n_alt={n_alternatives_spec}",
        leave=False,
    ):
        x = _sample_training_batch(batch_size, device, sampler)
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        _sync()
        t0 = time.perf_counter()
        logits = model(inp)
        _sync()
        t1 = time.perf_counter()
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            tgt.reshape(B * T),
            reduction="sum",
        )

        optimizer_layers.zero_grad(set_to_none=True)
        optimizer_embedder.zero_grad(set_to_none=True)
        _sync()
        t2 = time.perf_counter()
        loss.backward()
        _sync()
        t3 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_layers.step()
        optimizer_embedder.step()
        _sync()
        t4 = time.perf_counter()

        forward_ms_list.append((t1 - t0) * 1000.0)
        backward_ms_list.append((t3 - t2) * 1000.0)
        optimizer_step_ms_list.append((t4 - t3) * 1000.0)

    elapsed_s = sum(forward_ms_list) / 1000.0 + sum(backward_ms_list) / 1000.0 + sum(optimizer_step_ms_list) / 1000.0
    print(f"[DEBUG] timed loop finished, elapsed={elapsed_s:.3f}s", flush=True)

    native_stats = None
    if backend.device == "cuda" and backend.custom_kernel_mode != "none":
        native_stats = _get_native_lutorch_stats()

    def _mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    result: Dict[str, object] = {
        "backend_name": backend.name,
        "device": backend.device,
        "smooth_mode": smooth_mode,
        "n_alternatives_spec": n_alternatives_spec,
        "n_alternatives": n_alternatives,
        "untrained_loss": untrained_loss,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "profile_steps": profile_steps,
        "forward_ms_mean": _mean(forward_ms_list),
        "backward_ms_mean": _mean(backward_ms_list),
        "optimizer_step_ms_mean": _mean(optimizer_step_ms_list),
        "elapsed_s": elapsed_s,
        "native_lutorch_stats": native_stats,
    }
    return result


def _build_backend_grid() -> List[BackendConfig]:
    if torch.cuda.is_available():
        return [
            BackendConfig(
                name="gpu_pure",
                device="cuda",
                use_torch_compile=False,
                custom_kernel_mode="none",
            ),
            BackendConfig(
                name="gpu_compiled_no_k",
                device="cuda",
                use_torch_compile=True,
                custom_kernel_mode="none",
            ),
            BackendConfig(
                name="gpu_compiled_anchor",
                device="cuda",
                use_torch_compile=True,
                custom_kernel_mode="anchor_only",
            ),
            BackendConfig(
                name="gpu_compiled_all",
                device="cuda",
                use_torch_compile=True,
                custom_kernel_mode="all",
            ),
        ]
    print("[INFO] CUDA not available; running CPU backends only.")
    return [
        BackendConfig(
            name="cpu_pure",
            device="cpu",
            use_torch_compile=False,
            custom_kernel_mode="none",
        ),
    ]


def _write_markdown(results: List[Dict[str, object]], path: str) -> None:
    lines: List[str] = []
    lines.append("# LUTorch Transformer Profiling")
    lines.append("")
    lines.append("Wall-clock mean time per step (ms); on CUDA, `torch.cuda.synchronize()` is used so times reflect GPU work.")
    lines.append("")
    lines.append("| backend | smooth | n_alt | forward_ms | backward_ms | optimizer_step_ms | elapsed_s |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for res in results:
        lines.append(
            "| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.3f} |".format(
                res["backend_name"],
                res["smooth_mode"],
                res["n_alternatives_spec"],
                res["forward_ms_mean"],
                res["backward_ms_mean"],
                res["optimizer_step_ms_mean"],
                res["elapsed_s"],
            )
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-run details (native profiler when present)")
    lines.append("")

    for res in results:
        lines.append(
            f"### Backend `{res['backend_name']}`, smooth={res['smooth_mode']}, n_alt={res['n_alternatives_spec']}"
        )
        lines.append("")
        lines.append(f"- device: `{res['device']}`, batch_size: {res['batch_size']}, profile_steps: {res['profile_steps']}")
        lines.append(f"- forward_ms: {res['forward_ms_mean']:.2f}, backward_ms: {res['backward_ms_mean']:.2f}, optimizer_step_ms: {res['optimizer_step_ms_mean']:.2f}, elapsed_s: {res['elapsed_s']:.3f}")
        lines.append(f"- untrained_loss: {res['untrained_loss']:.6f}")
        lines.append("")
        native_stats = res.get("native_lutorch_stats")
        if native_stats:
            lines.append("**Native LUTorchManager profiling stats:**")
            lines.append("")
            lines.append("```")
            lines.append(str(native_stats))
            lines.append("```")
            lines.append("")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Wrote profiling results to {path}")


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    backends = _build_backend_grid()
    smooth_modes = [True, False]
    n_alt_specs = ["1", "3", "all"]

    all_results: List[Dict[str, object]] = []

    for backend in backends:
        for smooth in smooth_modes:
            for n_alt in n_alt_specs:
                print(
                    f"\n[RUN] backend={backend.name}, "
                    f"smooth_mode={smooth}, n_alternatives_spec={n_alt}"
                )
                # Use a much smaller workload on CPU so profiling is fast.
                if backend.device == "cpu":
                    batch_size = 2
                    warmup_steps = 2
                    profile_steps = 20
                else:
                    batch_size = 128
                    warmup_steps = 10
                    profile_steps = 100

                res = run_single_configuration(
                    backend=backend,
                    smooth_mode=smooth,
                    n_alternatives_spec=n_alt,
                    batch_size=batch_size,
                    warmup_steps=warmup_steps,
                    profile_steps=profile_steps,
                )
                all_results.append(res)

    # Place the markdown file alongside this script so it stays local to the
    # LUTorch tests rather than in the shared workbooks directory.
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "lutorch_profile_results.md")
    os.makedirs(out_dir, exist_ok=True)
    _write_markdown(all_results, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

