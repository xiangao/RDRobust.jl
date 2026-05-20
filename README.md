# RDRobust.jl

`RDRobust.jl` is a Julia implementation of the main `rdrobust` workflow for
regression discontinuity designs. It follows the estimators and bandwidth
selectors developed by Calonico, Cattaneo, Farrell, and Titiunik.

## What is included

- **`rdrobust`**: Local polynomial RD point estimators with robust bias-corrected confidence intervals.
- **`rdbwselect`**: Data-driven bandwidth selectors for RD designs.
- **`rdplot`**: Data-driven RD plots (returns data frames for plotting).

Supports:
- Local linear, local quadratic, and higher-order polynomial fits.
- MSE-optimal and CER-optimal bandwidth selection.
- Covariate adjustment.
- Cluster-robust inference.
- Nearest neighbor variance estimation.
- Fuzzy RD designs.
- Multiple kernels: Triangular (default), Epanechnikov, and Uniform.

## Installation

```julia
using Pkg
Pkg.add(path="/path/to/RDRobust.jl")
```

## Quick Start

```julia
using RDRobust
using CSV
using DataFrames

# Load data
df = CSV.read("rdrobust_senate.csv", DataFrame)
y = df.vote
x = df.margin

# RD estimation
results = rdrobust(y, x)

# Bandwidth selection
bw = rdbwselect(y, x)

# RD plot data
plot_data = rdplot(y, x)
```

## References

- Calonico, Cattaneo and Titiunik (2014): [Robust Data-Driven Inference in the Regression-Discontinuity Design](https://rdpackages.github.io/references/Calonico-Cattaneo-Titiunik_2014_Stata.pdf). *Stata Journal* 14(4): 909-946.
- Calonico, Cattaneo and Titiunik (2015): [rdrobust: An R Package for Robust Nonparametric Inference in Regression-Discontinuity Designs](https://rdpackages.github.io/references/Calonico-Cattaneo-Titiunik_2015_R.pdf). *R Journal* 7(1): 38-51.
- Calonico, Cattaneo, Farrell and Titiunik (2017): [rdrobust: Software for Regression Discontinuity Designs](https://rdpackages.github.io/references/Calonico-Cattaneo-Farrell-Titiunik_2017_Stata.pdf). *Stata Journal* 17(2): 372-404.
