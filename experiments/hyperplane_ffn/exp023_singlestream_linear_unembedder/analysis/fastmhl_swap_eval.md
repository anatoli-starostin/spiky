FastMHL frozen-swap of trained exp023 (single-stream + Linear).
Baseline hyperplane-reload (0 missing/0 unexpected, eval_steps=20 fresh val): 1.2188 bpb.
FastMHL frozen swap (tables transferred, 36 hyperplane addressing params dropped): 4.7708 bpb.
Delta vs baseline-same-protocol +3.552; vs exp023 reported 1.2063 +3.564; vs champion 1.1940 +3.577.
Degenerate: FastMHL fixed anchor-pair addressing is unrelated to exp023 learned dense hyperplanes; tables read at meaningless cells. Frozen only, no finetune.
