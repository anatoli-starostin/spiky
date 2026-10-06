"""Deployment export/load (PR 1): exact pow2 decompress-fold, export->load->forward parity, and the
guarantee that NO fp32 LUT table Parameter is allocated in the loaded inference model.

CPU + fp32: the int8 read (_pow2.int_blend_read) and the quant eval fake-quant path are both pure
torch, so this needs no CUDA / no JIT kernel."""
import json

import pytest
import torch
import torch.nn as nn

from spiky.lutorch_ex import (
    LUTSpec, QuantisedConfidenceLUT, ProjectionMHL,
    export_deployment, load_deployment, DeployedQuantisedConfidenceLUT,
)
from spiky.lutorch_ex.cartridges import _pow2

N_EMBD = 48          # == h*d_inner so compress/decompress are square-ish; champion-shaped but tiny tph
SPEC = dict(h_in=4, h_out=4, tph=8, nap=6, d_in=12, d_out=12)   # out_features = 4*12 = 48 = n_embd


def _spec():
    return LUTSpec(**SPEC)


class ToyModel(nn.Module):
    """A backbone Linear + two quant ProjectionMHL FFNs (residual) — exercises FFN compaction AND the
    backbone state_dict materialisation path."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(N_EMBD, N_EMBD)
        common = dict(quant_mode="p2_int8", weight_init_std=1e-2, read_top_n=2,
                      beta_init=2.0, gamma_init=1.0, read_tau_init=0.5, read_tau_learnable=True)
        self.ff0 = ProjectionMHL(QuantisedConfidenceLUT(_spec(), seed=1, **common), d_model=N_EMBD)
        self.ff1 = ProjectionMHL(QuantisedConfidenceLUT(_spec(), seed=2, **common), d_model=N_EMBD)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.ff0(x)
        x = x + self.ff1(x)
        return x


def _make_skeleton():
    with torch.device("meta"):
        return ToyModel()


def _trained_model():
    torch.manual_seed(0)
    m = ToyModel().eval()
    # give the tables some non-trivial values (as if trained)
    with torch.no_grad():
        for ff in (m.ff0, m.ff1):
            ff.cartridge.weights.normal_(0.0, 1e-2)
    return m


def test_pow2_decompress_fold_is_exact():
    m = _trained_model()
    ff = m.ff0
    scale = ff.cartridge.to_deployment()["decompress_scale"]        # 2^(e-6), [out_features]
    folded = ff.decompress.weight.detach() * scale.reshape(1, -1)
    # folding a power-of-two per-column scale is exact: dividing back recovers the original bit-for-bit
    recovered = folded / scale.reshape(1, -1)
    assert torch.equal(recovered, ff.decompress.weight.detach()), "pow2 fold/unfold not bit-exact"
    # and the scale is exactly a power of two (log2 is integral)
    log2 = torch.log2(scale)
    assert torch.allclose(log2, log2.round(), atol=0), "decompress_scale is not a pure power of two"


def test_export_load_forward_parity(tmp_path):
    m = _trained_model()
    x = torch.randn(16, N_EMBD)
    with torch.no_grad():
        y_ref = m(x)
    path = str(tmp_path / "deploy.lxq")
    export_deployment(m, path)
    dep = load_deployment(path, _make_skeleton, device="cpu")
    with torch.no_grad():
        y_dep = dep(x)
    assert y_dep.shape == y_ref.shape
    # int8 read reproduces the fake-quant eval value to quantisation round-off
    rel = (y_dep - y_ref).norm() / y_ref.norm().clamp_min(1e-30)
    assert rel < 1e-2, f"deployed vs trained output rel diff too large: {rel.item():.3e}"


def test_no_fp32_lut_table_after_load(tmp_path):
    m = _trained_model()
    path = str(tmp_path / "deploy.lxq")
    export_deployment(m, path)
    dep = load_deployment(path, _make_skeleton, device="cpu")
    for name, mod in dep.named_modules():
        if isinstance(mod, DeployedQuantisedConfidenceLUT):
            # no fp32 `weights` Parameter anywhere in the deployed cartridge
            assert "weights" not in dict(mod.named_parameters()), f"{name}: fp32 weights Parameter present"
            assert not any(p.is_floating_point() and p.numel() > mod.spec.d_out * 4 for p in mod.parameters()), \
                f"{name}: unexpected large fp32 Parameter"
            assert mod.packed.dtype == torch.int8, f"{name}: packed table not int8"
            assert not torch.is_floating_point(mod.packed)
    # the big tensor resident in each deployed FFN cartridge is the int8 packed table, not an fp32 table
    deployed = [mod for _, mod in dep.named_modules() if isinstance(mod, DeployedQuantisedConfidenceLUT)]
    assert len(deployed) == 2


# ---- serialisation: the loader follows the file's format, whatever is installed --------------------

def _payload():
    return {"a": torch.arange(6, dtype=torch.int8).reshape(2, 3), "b": torch.randn(4)}, {"format_version": 1, "x": [1, 2]}


def _assert_same(got, want):
    (gt, gm), (wt, wm) = got, want
    assert gm == wm and gt.keys() == wt.keys()
    assert all(torch.equal(gt[k], wt[k]) and gt[k].dtype == wt[k].dtype for k in wt)


def test_torch_format_file_loads_with_or_without_safetensors(tmp_path):
    """A torch.save payload (written where safetensors could not write) must load wherever it is read,
    including where safetensors IS installed -- the loader must not assume the writer's format."""
    from spiky.lutorch_ex.deploy import _load_payload
    tensors, meta = _payload()
    path = str(tmp_path / "torch_format.lxq")
    torch.save({"tensors": tensors, "meta_json": json.dumps(meta)}, path)
    _assert_same(_load_payload(path), (tensors, meta))


def test_save_falls_back_when_safetensors_cannot_write(tmp_path, monkeypatch):
    """safetensors importable but unable to write (e.g. numpy missing: ModuleNotFoundError inside save_file):
    the payload is written with torch.save and loads back unchanged."""
    st = pytest.importorskip("safetensors.torch")
    from spiky.lutorch_ex.deploy import _load_payload, _save_payload, _ZIP_MAGIC

    def no_numpy(*a, **k):
        raise ModuleNotFoundError("No module named 'numpy'")
    monkeypatch.setattr(st, "save_file", no_numpy)
    tensors, meta = _payload()
    path = str(tmp_path / "fallback.lxq")
    _save_payload(path, tensors, meta)
    assert open(path, "rb").read(4) == _ZIP_MAGIC
    _assert_same(_load_payload(path), (tensors, meta))


def test_payload_roundtrip(tmp_path):
    """Whatever this environment writes, it reads back."""
    from spiky.lutorch_ex.deploy import _load_payload, _save_payload
    tensors, meta = _payload()
    path = str(tmp_path / "roundtrip.lxq")
    _save_payload(path, tensors, meta)
    _assert_same(_load_payload(path), (tensors, meta))
