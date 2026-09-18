# Corrected train_bpb figures (data-weighted divisor 4.747)

Regenerated after finding the earlier train_bpb plots used a WRONG bytes/token divisor of
6.575 (the vocab-mean over the 32,768 vocab entries). The correct DATA-WEIGHTED divisor is
**4.747 bytes/token** (measured on the 4-shard ClimbMix train corpus: 1,011,037,744 UTF-8
text bytes / ~213M tokens over 340,992 documents). Formula: `bpb = loss_nats / (ln2 * 4.747)`.

- val_bpb is UNAFFECTED: tools/fixed_eval.evaluate_bpb_fixed computes it as
  `total_nats / (ln2 * sum_of_actual_token_byte_lengths)` -- it never used any fixed divisor.
- Only train_bpb (the nats-loss -> bpb conversion) changes: it is 6.575/4.747 = 1.385x LARGER
  than the earlier figures, so the true train-val gaps are ~0.05-0.11 (not ~0.35).

Final numbers (final train_bpb 6.575 -> 4.747 | val unchanged | gap 6.575 -> 4.747):
- vanilla48k       0.7622 -> 1.0557 | val 1.115420 | gap 0.3532 -> 0.0597
- p2_int8 tph64_h8 0.7479 -> 1.0359 | val 1.120429 | gap 0.3726 -> 0.0846
- noquant tph64_h8 0.7435 -> 1.0298 | val 1.122858 | gap 0.3794 -> 0.0931
- bs64 tph128      0.7295 -> 1.0105 | val 1.119680 | gap 0.3902 -> 0.1092
- headdrop float   0.7622 -> 1.0557 | val 1.107921 | gap 0.3457 -> 0.0522

Cross-run ORDERING is unchanged by the divisor (it cancels in gap differences).
