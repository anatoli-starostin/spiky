# exp_g_0041 -- never run, deliberately dropped

Was to be the lambda=10 rung of the sparsity-penalty ladder.  Dropped before launch because exp_g_0040 (lambda 50) had already shown the ladder answers nothing: it reproduced exp_g_0039's (lambda 100) density trajectory to five decimal places (step 200: frac_zero 0.831469 vs 0.831464; step 400: 42.0165 vs 42.0200 non-zeros per hyperplane) while its penalty VALUE was exactly halved.

AdamW divides each parameter's update by that parameter's own running gradient magnitude, so wherever the penalty gradient dominates, lambda cancels in m/sqrt(v) and the update is ~lr regardless of lambda's size.  Lambda is close to a no-op knob over this range; the density reached is set by the surrogate's fixed point and the LR.  lambda=10 would have landed on the same curve.

**The number 0041 is left as a deliberate gap.**  The work continues at exp_g_0042, which replaces the one-way push with a target-density hinge -- relu(surrogate - 64/384) -- so the pressure stops once the wanted sparsity is reached instead of pushing forever.
