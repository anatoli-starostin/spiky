"""Shared config + fixed seeded eval set for the DB-backed neuroevolution framework
(GitHub issue #81). Nothing here holds secrets; the Mongo URI is read from the env."""
import os
import random

import neuroevo_lut as N

# ---- Mongo ----
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("NE_DB", "neuroevo")
COLLECTION = os.environ.get("NE_COLLECTION", "genomes")

# ---- genome states ----
NEW_BORN = "NEW_BORN"
SCORED = "SCORED"
PROCESSED = "PROCESSED"
CLAIMED = "CLAIMED"          # transient: a worker has stamped the doc mid-claim

# ---- pack sizing (see #30: native 512 neuron-meta cap, ~128-genome practical pack) ----
PACK_LIMIT = int(os.environ.get("NE_PACK_LIMIT", 128))     # max genomes per built net
META_SOFT_CAP = int(os.environ.get("NE_META_SOFT_CAP", 480))  # stop before the hard 512
DROP_ON_OVERSHOOT = int(os.environ.get("NE_DROP_ON_OVERSHOOT", 16))  # genomes to release on build error

# ---- Scorer / Generator batch sizes ----
GENERATOR_PARENTS = int(os.environ.get("NE_GEN_PARENTS", 64))   # top SCORED claimed per Generator iter
GENERATOR_OFFSPRING = int(os.environ.get("NE_GEN_OFFSPRING", 128))
SEED_POP = int(os.environ.get("NE_SEED_POP", 256))              # bootstrap NEW_BORN if empty

# ---- claim lifecycle ----
CLAIM_TIMEOUT_S = float(os.environ.get("NE_CLAIM_TIMEOUT_S", 120))

# ---- device for scoring ----
DEVICE = os.environ.get("NE_DEVICE", "cpu")

# ---- fixed seeded eval set (comparable, reproducible scores across async scoring) ----
EVAL_SEED = int(os.environ.get("NE_EVAL_SEED", 20260730))
EVAL_N = int(os.environ.get("NE_EVAL_N", 64))
EVAL_VERSION = "lut-v1-seed%d-n%d" % (EVAL_SEED, EVAL_N)


def fixed_eval_set():
    """The ONE frozen eval set used for every genome, so stored scores are comparable.
    Returns (xs, true_orders). Deterministic in EVAL_SEED/EVAL_N (-> EVAL_VERSION)."""
    rng = random.Random(EVAL_SEED)
    xs = [[rng.uniform(-1, 1) for _ in range(N.D)] for _ in range(EVAL_N)]
    tos = [N.oracle_order(x) for x in xs]
    return xs, tos
