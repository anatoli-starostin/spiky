"""Tests for cage_policy.classify — the opt-in tailnet network mode and that the
default-deny cage is otherwise unchanged.

`sbox --net tailnet -- <cmd>` is a single fixed, well-understood mode and is
greenlit; anything else after `--net` (unknown mode, raw host:port, malformed)
is fail-closed -> gated. Bare `sbox` (no network) stays green.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import cage_policy  # noqa: E402


def g(cmd):
    return cage_policy.classify("Bash", {"command": cmd})


# ── bare sbox (no network) stays green, unchanged ────────────────────────────
def test_bare_sbox_still_green():
    assert g("sbox python train.py") == "green"
    assert g("sbox bash -c 'cd ~/projects/x && python t.py | tee /tmp/log'") == "green"


# ── the one opt-in mode -> green ─────────────────────────────────────────────
def test_net_tailnet_green():
    assert g("sbox --net tailnet -- python train.py") == "green"
    assert g("sbox --net tailnet -- wandb sync") == "green"


# ── any other --net value -> gated (fail-closed) ─────────────────────────────
def test_net_unknown_value_gated():
    assert g("sbox --net internet -- python x") == "gated"
    assert g("sbox --net all -- curl x") == "gated"
    assert g("sbox --net Tailnet -- x") == "gated"          # case-sensitive


def test_net_raw_endpoint_gated():
    assert g("sbox --net 100.64.0.1:8080 -- curl x") == "gated"
    assert g("sbox --net host.example:8080 -- x") == "gated"


# ── malformed --net shapes -> gated ──────────────────────────────────────────
def test_net_malformed_gated():
    assert g("sbox --net tailnet python train.py") == "gated"  # missing '--'
    assert g("sbox --net tailnet --") == "gated"               # no command after '--'
    assert g("sbox --net -- python x") == "gated"              # mode slot is '--'
    assert g("sbox --net") == "gated"                          # nothing after flag


# ── --net appearing as the caged program's own arg -> bare green (no net) ────
def test_net_as_inner_arg_is_bare_green():
    assert g("sbox myprog --net tailnet") == "green"


# ── default-deny: raw network / installs / containers still gate ─────────────
def test_raw_network_still_gated():
    assert g("curl http://example.com") == "gated"
    assert g("pip install wandb") == "gated"
    assert g("wandb login --host http://x KEY") == "gated"


# ── operators/chaining still gate even with a valid --net tailnet prefix ─────
def test_operators_gate_even_with_net_tailnet():
    assert g("sbox --net tailnet -- python x && ls") == "gated"          # multi-segment
    assert g("sbox --net tailnet -- python x > /home/u/out") == "gated"  # redirect to real path
    assert g("sbox --net tailnet -- python $(cat x)") == "gated"         # command substitution


# ── existing green behavior preserved ────────────────────────────────────────
def test_existing_green_preserved():
    assert g("git -C ~/projects/spiky status") == "green"
    assert g("ls ~/projects") == "green"
    assert cage_policy.classify("Read", {"file_path": "/etc/hosts"}) == "ungated"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
