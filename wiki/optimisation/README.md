# 2. Optimisation techniques

How LUT parameters are trained: which standard optimisers work, what helps training speed and stability, which regularisers and normalisations help, and the idea of optimisers designed specifically for LUTs. The pages are short notes on techniques we tried, one subsection per technique; evaluation numbers are kept short and link to the [Architectures](../architectures/README.md) pages, where the runs are described.

- [Basic optimisers applied to LUTs](basic-optimisers.md): AdamW, Lion, Signum, SGD, momentum, Muon and its table analogue
- [Training efficiency](training-efficiency.md): batch size, schedules, initialisation, precision, wall clock, soft to hard, distillation
- [Normalisation](normalisation.md): layer norm and QK norm, the norm before the tables, output-side normalisations
- [Regularisation](regularisation.md): table dropout, total-variation penalties, flip and bf16 noise, weight decay
- [Special LUT optimisers](special-optimisers.md): bit and K-bit optimisers, row-visit-aware variants, one open idea
