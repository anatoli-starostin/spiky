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

# ---- diversity pressure (LOCAL rules, approximate global effect; no barriers) ----
# (1) IMMIGRANTS: the generator injects fresh random_genome() NEW_BORN as a steady rate, as a
#     fraction of the offspring it bred that iteration. 0.5 => ~1 immigrant per 2 offspring.
#     Set 0.0 to disable (legacy: no immigrants). Immigrants get a fresh lineage_id + depth 0.
IMMIGRANT_RATIO = float(os.environ.get("NE_IMMIGRANT_RATIO", 0.5))
# (2) SCORER INTAKE decoupled from score: each scorer batch is claimed FIFO (insertion order,
#     via _id) with prob (1-EPS), or by priority with prob EPS. EPS=1.0 => pure priority
#     (legacy behavior); EPS=0.0 => pure FIFO. Breaks the score->intake->score collapse loop.
SCORE_PRIORITY_EPS = float(os.environ.get("NE_SCORE_PRIORITY_EPS", 0.1))
# (3) LINEAGE-CAPPED parent selection via over-fetch-and-filter (no global barrier): peek the
#     top-OVERFETCH SCORED by priority, then keep at most LINEAGE_CAP per lineage_id (root
#     ancestor), up to GENERATOR_PARENTS. A very large LINEAGE_CAP => no cap (legacy behavior).
GENERATOR_OVERFETCH = int(os.environ.get("NE_GEN_OVERFETCH", 400))
LINEAGE_CAP = int(os.environ.get("NE_LINEAGE_CAP", 3))
# Generator backlog THROTTLE: skip breeding while the unscored NEW_BORN backlog is at/above this
# ceiling, so scorers keep up and (nearly) every created genome actually gets evaluated instead of
# piling up unscored. 0 = disabled (legacy: breed every iteration regardless of backlog).
MAX_BACKLOG = int(os.environ.get("NE_MAX_BACKLOG", 0))

# ---- device for scoring ----
DEVICE = os.environ.get("NE_DEVICE", "cpu")

# ---- fitness metric for the DB scorer (stored as `score`, which becomes SCORED priority) ----
# "taub" (DEFAULT) = Kendall tau-b ONLY — dense, ties-aware, all-fire-gated; the SOLE objective.
# "composite" = legacy 0.5*strict + 0.5*order (from N._score_population). Switch via NE_FITNESS.
FITNESS = os.environ.get("NE_FITNESS", "taub")

# ---- fixed seeded eval set (comparable, reproducible scores across async scoring) ----
EVAL_SEED = int(os.environ.get("NE_EVAL_SEED", 20260730))
EVAL_N = int(os.environ.get("NE_EVAL_N", 512))
EVAL_VERSION = "lut-v1-seed%d-n%d" % (EVAL_SEED, EVAL_N)


def fixed_eval_set():
    """The ONE frozen eval set used for every genome, so stored scores are comparable.
    Returns (xs, true_orders). Deterministic in EVAL_SEED/EVAL_N (-> EVAL_VERSION)."""
    rng = random.Random(EVAL_SEED)
    xs = [[rng.uniform(-1, 1) for _ in range(N.D)] for _ in range(EVAL_N)]
    tos = [N.oracle_order(x) for x in xs]
    return xs, tos
