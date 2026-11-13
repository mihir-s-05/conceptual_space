# Control: recurrent depth

Known formulation change to `c.fixed_depth` and `c.unshared` (alternative `alt.recurrent_depth`;
Geiping et al. 2025, arXiv:2502.05171). Compute per token matches the 6-layer baseline
(6 block applications) with about half the non-embedding parameters.

Role in the calibration cohort: a reformulation whose short-run and long-run behaviour we measure
directly, to see whether the screening rule rejects it early.
