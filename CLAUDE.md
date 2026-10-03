# RDRobust.jl

Julia port of `rdrobust` (Calonico, Cattaneo, Farrell and Titiunik): `rdrobust`,
`rdbwselect`, `rdplot`. Source in `src/` (`rdrobust_impl.jl`, `rdbwselect.jl`,
`rdplot.jl`, `utils.jl`). Docs: Documenter, `docs/`.

## Validation against R

`test/runtests.jl` checks estimates, SEs, bandwidths and effective N against R
rdrobust 4.0.0 (`test/r_reference.csv`, written by `test/make_r_reference.R`):
sharp and fuzzy, with and without covariates, nn and hc1, to 1e-6. Run with
`julia --project=. -e 'using Pkg; Pkg.test()'` (about 2 minutes).

## Fixes 2026-10-03 (branch `covs-tests`)

- `rdbwselect` pilot bandwidth used the type-7 IQR; R uses `quantile(type = 2)`.
  Bandwidths differed from R by ~3e-5 relative for most sample sizes (the
  full Senate sample happened to agree). Now `quantile_type2` in `utils.jl`.
- `covs_drop_fun` now follows R: pivoted QR with relative tolerance 1e-7
  (was absolute 1e-5), kept columns in original order.
- `rdrobust` warns when the partialled-out covariate block is rank deficient.
  That happens when covariates are functions of the running variable over the
  mass points inside the bandwidth (Fetter mortgages: quarter-of-birth dummies
  with 3 quarters per side). The jump is then not identified. Julia's `pinv`
  and R's `ginv(tol = 1e-20)` return different arbitrary answers (R gave 0.096
  or 0.838 depending on whether a redundant intercept column was passed).
  This was not a port bug; do not try to make Julia "match" R in that case.
