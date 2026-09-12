"""Tests for cage_policy.classify — focused on the opt-in network allowlist and
that the default-deny cage is unchanged.

The net-allow greenlight is fail-closed: `sbox --net-allow NAME -- <argv>` is
green ONLY when NAME is a committed key in the allowlist file. We point
AGENT_CAGE_NET_ALLOWLIST at a temp fixture BEFORE importing cage_policy (it
reads the path from the env at import time).
"""
import atexit
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# committed-allowlist fixture: TESTNET and WANDB are the only greenlit names.
_fx = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
_fx.write("# a comment line\n"
          "TESTNET  example.com:443  cdn.example.com:443\n"
          "WANDB    host.example:8080\n")
_fx.flush(); _fx.close()
os.environ["AGENT_CAGE_NET_ALLOWLIST"] = _fx.name
atexit.register(lambda: os.unlink(_fx.name))

import cage_policy  # noqa: E402


def g(cmd):
    return cage_policy.classify("Bash", {"command": cmd})


# ── bare sbox (no network) stays green, unchanged ────────────────────────────
def test_bare_sbox_still_green():
    assert g("sbox python train.py") == "green"
    assert g("sbox bash -c 'cd ~/projects/x && python t.py | tee /tmp/log'") == "green"


# ── --net-allow with a COMMITTED name -> green ───────────────────────────────
def test_net_allow_committed_green():
    assert g("sbox --net-allow TESTNET -- python train.py") == "green"
    assert g("sbox --net-allow WANDB -- wandb sync") == "green"


# ── --net-allow with an UNKNOWN name -> gated (no self-grant) ────────────────
def test_net_allow_unknown_gated():
    assert g("sbox --net-allow NOPE -- python train.py") == "gated"
    assert g("sbox --net-allow wandb -- x") == "gated"   # case-sensitive; 'wandb' != 'WANDB'


# ── a raw host:port passed as the name -> gated ──────────────────────────────
def test_net_allow_raw_endpoint_gated():
    assert g("sbox --net-allow example.com:443 -- curl https://example.com") == "gated"
    assert g("sbox --net-allow host.example:8080 -- x") == "gated"


# ── malformed --net-allow shapes -> gated ────────────────────────────────────
def test_net_allow_malformed_gated():
    assert g("sbox --net-allow TESTNET python train.py") == "gated"  # missing '--'
    assert g("sbox --net-allow TESTNET --") == "gated"               # no command after '--'
    assert g("sbox --net-allow -- python x") == "gated"              # name slot is '--'
    assert g("sbox --net-allow") == "gated"                          # nothing after flag


# ── --net-allow appearing as the caged program's own arg -> bare green (no net) ─
def test_net_allow_as_inner_arg_is_bare_green():
    # sbox's own $1 is 'myprog', so no network is granted; it's a normal caged run.
    assert g("sbox myprog --net-allow foo") == "green"


# ── default-deny: raw network / installs / containers still gate ─────────────
def test_raw_network_still_gated():
    assert g("curl http://example.com") == "gated"
    assert g("pip install wandb") == "gated"
    assert g("podman run wandb/local") == "gated"
    assert g("wandb login --host http://x KEY") == "gated"


# ── operators/chaining still gate even with a valid --net-allow prefix ───────
def test_operators_gate_even_with_net_allow():
    assert g("sbox --net-allow TESTNET -- python x && ls") == "gated"          # multi-segment
    assert g("sbox --net-allow TESTNET -- python x > /home/u/out") == "gated"  # redirect to real path
    assert g("sbox --net-allow TESTNET -- python $(cat x)") == "gated"         # command substitution


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
