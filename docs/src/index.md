# RDRobust.jl

`RDRobust.jl` is a Julia implementation of the main `rdrobust` workflow for
regression discontinuity (RD) designs. It follows the estimators and
bandwidth selectors developed by Calonico, Cattaneo, Farrell, and Titiunik.

## What is included

- [`rdrobust`](@ref): local polynomial RD point estimators with robust
  bias-corrected confidence intervals.
- [`rdbwselect`](@ref): data-driven bandwidth selectors for RD designs.
- [`rdplot`](@ref): data-driven RD plots (returns data frames for plotting).

Supports local linear/quadratic/higher-order polynomial fits, MSE- and
CER-optimal bandwidth selection, covariate adjustment, cluster-robust
inference, nearest-neighbor variance estimation, fuzzy RD designs, and the
triangular/Epanechnikov/uniform kernels.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/xiangao/RDRobust.jl")
```

## Getting started

This example reproduces the canonical `rdrobust` senate-election
illustration (Lee, 2008): the running variable is a Democratic candidate's
margin of victory, the outcome is that party's vote share in the next
election, and the cutoff is a margin of zero (winning vs. losing the prior
election).

```@example senate
using RDRobust, CSV, DataFrames

df = CSV.read(joinpath(pkgdir(RDRobust), "docs", "data", "rdrobust_senate.csv"), DataFrame)
y, x = df.vote, df.margin

res = rdrobust(y, x)
res.Estimate[:, [:tau_us, :tau_bc]]
```

The conventional and robust bias-corrected estimates:

```@example senate
res.Estimate.tau_us[1], res.Estimate.tau_bc[1]
```

Bandwidth selection and RD-plot data work the same way:

```@example senate
bw = rdbwselect(y, x)
bw.bws[1, [:h_left, :h_right]]
```

```@example senate
rd = rdplot(y, x)
first(rd.vars_bins, 5)
```

## References

- Calonico, Cattaneo and Titiunik (2014): [Robust Data-Driven Inference in the Regression-Discontinuity Design](https://rdpackages.github.io/references/Calonico-Cattaneo-Titiunik_2014_Stata.pdf). *Stata Journal* 14(4): 909-946.
- Calonico, Cattaneo and Titiunik (2015): [rdrobust: An R Package for Robust Nonparametric Inference in Regression-Discontinuity Designs](https://rdpackages.github.io/references/Calonico-Cattaneo-Titiunik_2015_R.pdf). *R Journal* 7(1): 38-51.
- Calonico, Cattaneo, Farrell and Titiunik (2017): [rdrobust: Software for Regression Discontinuity Designs](https://rdpackages.github.io/references/Calonico-Cattaneo-Farrell-Titiunik_2017_Stata.pdf). *Stata Journal* 17(2): 372-404.
