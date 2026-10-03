# Reference values for test/runtests.jl, from R rdrobust 4.0.0.
# Run from the test directory: Rscript make_r_reference.R
suppressPackageStartupMessages({library(rdrobust); library(dplyr)})
sen <- read.csv("rdrobust_senate.csv")
set.seed(2013)
vet <- causaldata::mortgages %>% filter(abs(qob_minus_kw) < 12) %>%
  select(home_ownership, qob_minus_kw, vet_wwko, nonwhite) %>% slice_sample(n = 6000)
write.csv(vet, "mortgages_subset.csv", row.names = FALSE)
cs <- as.matrix(sen[, c("class", "termshouse", "termssenate")])
fits <- list(
  sharp        = rdrobust(sen$vote, sen$margin),
  sharp_covs   = rdrobust(sen$vote, sen$margin, covs = cs),
  sharp_hc1    = rdrobust(sen$vote, sen$margin, covs = cs, vce = "hc1"),
  fuzzy        = rdrobust(vet$home_ownership, vet$qob_minus_kw, fuzzy = vet$vet_wwko),
  fuzzy_covs   = rdrobust(vet$home_ownership, vet$qob_minus_kw, fuzzy = vet$vet_wwko,
                          covs = vet$nonwhite)
)
ref <- do.call(rbind, lapply(names(fits), function(k) { r <- fits[[k]]
  data.frame(case = k, tau_us = r$Estimate[1], tau_bc = r$Estimate[2], se_us = r$Estimate[3],
             se_rb = r$Estimate[4], h = r$bws[1, 1], b = r$bws[2, 1],
             N_h_l = r$N_h[1], N_h_r = r$N_h[2]) }))
write.csv(ref, "r_reference.csv", row.names = FALSE)
print(ref, digits = 10)
