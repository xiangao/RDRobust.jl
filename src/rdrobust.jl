module RDRobustEstimation

using ..Utils
using ..BandwidthSelection
using Missings
using Statistics
using DataFrames
using LinearAlgebra
using Distributions

export rdrobust, RDRobustOutput

struct RDRobustOutput
    Estimate::DataFrame
    bws::DataFrame
    coef::DataFrame
    se::DataFrame
    t::DataFrame
    pv::DataFrame
    ci::DataFrame
    beta_p_l::Matrix{Float64}
    beta_p_r::Matrix{Float64}
    V_cl_l::Matrix{Float64}
    V_cl_r::Matrix{Float64}
    V_rb_l::Matrix{Float64}
    V_rb_r::Matrix{Float64}
    N::Vector{Int}
    N_h::Vector{Int}
    N_b::Vector{Int}
    M::Vector{Int}
    tau_cl::Vector{Float64}
    tau_bc::Vector{Float64}
    c::Float64
    p::Int
    q::Int
    bias::Vector{Float64}
    kernel::String
    all_results::Bool
    vce::String
    bwselect::String
    level::Float64
    masspoints::String
end

function rdrobust(y, x; c=0.0, fuzzy=nothing, deriv=0, p=nothing, q=nothing,
                  h=nothing, b=nothing, rho=nothing, covs=nothing, covs_drop=true,
                  kernel="tri", weights=nothing, bwselect="mserd", vce="nn",
                  cluster=nothing, nnmatch=3, level=95.0, scalepar=1.0,
                  scaleregul=1.0, sharpbw=false, all_results=false, subset=nothing,
                  masspoints="adjust", bwcheck=nothing, bwrestrict=true,
                  stdvars=false)

    # Input cleaning
    # Convert to Float64 and handle Missing
    x = [ismissing(v) ? NaN : Float64(v) for v in reshape(x, :)]
    y = [ismissing(v) ? NaN : Float64(v) for v in reshape(y, :)]
    if !isnothing(fuzzy) fuzzy = [ismissing(v) ? NaN : Float64(v) for v in reshape(fuzzy, :)] end
    if !isnothing(cluster) cluster = [ismissing(v) ? NaN : Float64(v) for v in reshape(cluster, :)] end
    if !isnothing(weights) weights = [ismissing(v) ? NaN : Float64(v) for v in reshape(weights, :)] end
    if !isnothing(covs) covs = [ismissing(v) ? NaN : Float64(v) for v in covs] end
    
    if !isnothing(subset)
        x = x[subset]
        y = y[subset]
        if !isnothing(fuzzy) fuzzy = fuzzy[subset] end
        if !isnothing(cluster) cluster = cluster[subset] end
        if !isnothing(weights) weights = weights[subset] end
        if !isnothing(covs) covs = covs[subset, :] end
    end
    
    # x = Float64.(x)  # Already done above
    # y = Float64.(y)  # Already done above
    
    if isnothing(p) p = deriv + 1 end
    if isnothing(q) q = p + 1 end
    
    na_ok = complete_cases(x) .& complete_cases(y)
    if !isnothing(cluster) na_ok .&= complete_cases(cluster) end
    if !isnothing(covs) na_ok .&= complete_cases(covs) end
    if !isnothing(fuzzy) na_ok .&= complete_cases(fuzzy) end
    if !isnothing(weights) na_ok .&= complete_cases(weights) .& (weights .>= 0) end
    
    x = x[na_ok]
    y = y[na_ok]
    if !isnothing(covs) covs = covs[na_ok, :] end
    if !isnothing(fuzzy) fuzzy = fuzzy[na_ok] end
    if !isnothing(cluster) cluster = cluster[na_ok] end
    if !isnothing(weights) weights = weights[na_ok] end
    
    if vce == "nn" || masspoints in ["check", "adjust"]
        idx = sortperm(x)
        x = x[idx]
        y = y[idx]
        if !isnothing(covs) covs = covs[idx, :] end
        if !isnothing(fuzzy) fuzzy = fuzzy[idx] end
        if !isnothing(cluster) cluster = cluster[idx] end
        if !isnothing(weights) weights = weights[idx] end
    end
    
    N_l = count(x .< c)
    N_r = count(x .>= c)
    N = N_l + N_r
    
    X_l = x[x .< c]
    X_r = x[x .>= c]
    Y_l = y[x .< c]
    Y_r = y[x .>= c]
    
    M_l = N_l
    M_r = N_r
    X_uniq_l = Float64[]
    X_uniq_r = Float64[]
    if masspoints in ["check", "adjust"]
        X_uniq_l = sort(unique(X_l), rev=true)
        X_uniq_r = sort(unique(X_r))
        M_l = length(X_uniq_l)
        M_r = length(X_uniq_r)
    end
    
    # Collinearity check for covariates
    dZ = 0
    if !isnothing(covs)
        if covs_drop
            covs = covs_drop_fun(covs)
        end
        dZ = size(covs, 2)
    end
    
    # Bandwidth selection
    h_l = h_r = b_l = b_r = 0.0
    if isnothing(h)
        rdbws = rdbwselect(y, x, c=c, fuzzy=fuzzy, deriv=deriv, p=p, q=q,
                           covs=covs, covs_drop=covs_drop, kernel=kernel, weights=weights,
                           bwselect=bwselect, vce=vce, cluster=cluster, nnmatch=nnmatch,
                           scaleregul=scaleregul, sharpbw=sharpbw, masspoints=masspoints,
                           bwcheck=bwcheck, bwrestrict=bwrestrict, stdvars=stdvars)
        h_l = rdbws.bws[1, :h_left]
        h_r = rdbws.bws[1, :h_right]
        b_l = rdbws.bws[1, :b_left]
        b_r = rdbws.bws[1, :b_right]
        if !isnothing(rho)
            b_l = h_l / rho
            b_r = h_r / rho
        end
    else
        if h isa Real
            h_l = h_r = Float64(h)
        else
            h_l, h_r = Float64.(h)
        end
        if isnothing(b)
            if isnothing(rho)
                b_l, b_r = h_l, h_r
            else
                b_l, b_r = h_l / rho, h_r / rho
            end
        else
            if b isa Real
                b_l = b_r = Float64(b)
            else
                b_l, b_r = Float64.(b)
            end
        end
    end
    
    # Weighting and indicator functions
    w_h_l = rdrobust_kweight(X_l, c, h_l, kernel)
    w_h_r = rdrobust_kweight(X_r, c, h_r, kernel)
    w_b_l = rdrobust_kweight(X_l, c, b_l, kernel)
    w_b_r = rdrobust_kweight(X_r, c, b_r, kernel)
    
    if !isnothing(weights)
        fw_l = weights[x .< c]
        fw_r = weights[x .>= c]
        w_h_l .*= fw_l
        w_h_r .*= fw_r
        w_b_l .*= fw_l
        w_b_r .*= fw_r
    end
    
    ind_h_l = w_h_l .> 0
    ind_h_r = w_h_r .> 0
    ind_b_l = w_b_l .> 0
    ind_b_r = w_b_r .> 0
    
    N_h_l = sum(ind_h_l)
    N_h_r = sum(ind_h_r)
    N_b_l = sum(ind_b_l)
    N_b_r = sum(ind_b_r)
    
    ind_l = h_l > b_l ? ind_h_l : ind_b_l
    ind_r = h_r > b_r ? ind_h_r : ind_b_r
    
    eN_l = sum(ind_l)
    eN_r = sum(ind_r)
    eY_l = Y_l[ind_l]
    eY_r = Y_r[ind_r]
    eX_l = X_l[ind_l]
    eX_r = X_r[ind_r]
    W_h_l = w_h_l[ind_l]
    W_h_r = w_h_r[ind_r]
    W_b_l = w_b_l[ind_l]
    W_b_r = w_b_r[ind_r]
    
    dups_l = zeros(Int, eN_l)
    dupsid_l = zeros(Int, eN_l)
    dups_r = zeros(Int, eN_r)
    dupsid_r = zeros(Int, eN_r)
    
    if vce == "nn"
        for i in 1:eN_l
            d = count(==(eX_l[i]), eX_l)
            dups_l[i] = d
            first_idx = findfirst(==(eX_l[i]), eX_l)
            dupsid_l[i] = i - first_idx + 1
        end
        for i in 1:eN_r
            d = count(==(eX_r[i]), eX_r)
            dups_r[i] = d
            first_idx = findfirst(==(eX_r[i]), eX_r)
            dupsid_r[i] = i - first_idx + 1
        end
    end
    
    u_l = (eX_l .- c) ./ h_l
    u_r = (eX_r .- c) ./ h_r
    
    R_q_l = fill(NaN, eN_l, q + 1)
    R_q_r = fill(NaN, eN_r, q + 1)
    for j in 0:q
        R_q_l[:, j + 1] = (eX_l .- c).^j
        R_q_r[:, j + 1] = (eX_r .- c).^j
    end
    R_p_l = R_q_l[:, 1:(p+1)]
    R_p_r = R_q_r[:, 1:(p+1)]
    
    # Estimation matrices
    L_l = (R_p_l .* reshape(W_h_l, :, 1))' * (u_l.^(p + 1))
    L_r = (R_p_r .* reshape(W_h_r, :, 1))' * (u_r.^(p + 1))
    
    invG_q_l = qrXXinv(R_q_l .* sqrt.(reshape(W_b_l, :, 1)))
    invG_q_r = qrXXinv(R_q_r .* sqrt.(reshape(W_b_r, :, 1)))
    invG_p_l = qrXXinv(R_p_l .* sqrt.(reshape(W_h_l, :, 1)))
    invG_p_r = qrXXinv(R_p_r .* sqrt.(reshape(W_h_r, :, 1)))
    
    e_p1 = zeros(q + 1)
    e_p1[p + 2] = 1.0 # bias term position
    
    Q_q_l = ((R_p_l .* reshape(W_h_l, :, 1))' .- (h_l^(p + 1) .* L_l * e_p1') * (invG_q_l * R_q_l') .* reshape(W_b_l, 1, :))'
    Q_q_r = ((R_p_r .* reshape(W_h_r, :, 1))' .- (h_r^(p + 1) .* L_r * e_p1') * (invG_q_r * R_q_r') .* reshape(W_b_r, 1, :))'
    
    D_l = reshape(eY_l, :, 1)
    D_r = reshape(eY_r, :, 1)
    
    dT = 0
    if !isnothing(fuzzy)
        dT = 1
        eT_l = fuzzy[x .< c][ind_l]
        eT_r = fuzzy[x .>= c][ind_r]
        D_l = hcat(D_l, eT_l)
        D_r = hcat(D_r, eT_r)
    end
    
    if !isnothing(covs)
        eZ_l = covs[x .< c, :][ind_l, :]
        eZ_r = covs[x .>= c, :][ind_r, :]
        D_l = hcat(D_l, eZ_l)
        D_r = hcat(D_r, eZ_r)
    end
    
    beta_p_l = invG_p_l * ((R_p_l .* reshape(W_h_l, :, 1))' * D_l)
    beta_q_l = invG_q_l * ((R_q_l .* reshape(W_b_l, :, 1))' * D_l)
    beta_bc_l = invG_p_l * (Q_q_l' * D_l)
    
    beta_p_r = invG_p_r * ((R_p_r .* reshape(W_h_r, :, 1))' * D_r)
    beta_q_r = invG_q_r * ((R_q_r .* reshape(W_b_r, :, 1))' * D_r)
    beta_bc_r = invG_p_r * (Q_q_r' * D_r)
    
    beta_p = beta_p_r .- beta_p_l
    beta_bc = beta_bc_r .- beta_bc_l
    
    # Scaling and results
    fact_deriv = factorial(deriv)
    tau_cl = 0.0
    tau_bc = 0.0
    bias_l = 0.0
    bias_r = 0.0
    s_Y = [1.0]
    
    if isnothing(covs)
        tau_cl = scalepar * fact_deriv * beta_p[deriv + 1, 1]
        tau_bc = scalepar * fact_deriv * beta_bc[deriv + 1, 1]
        
        tau_Y_cl_l = scalepar * fact_deriv * beta_p_l[deriv + 1, 1]
        tau_Y_cl_r = scalepar * fact_deriv * beta_p_r[deriv + 1, 1]
        tau_Y_bc_l = scalepar * fact_deriv * beta_bc_l[deriv + 1, 1]
        tau_Y_bc_r = scalepar * fact_deriv * beta_bc_r[deriv + 1, 1]
        bias_l = tau_Y_cl_l - tau_Y_bc_l
        bias_r = tau_Y_cl_r - tau_Y_bc_r
        
        if !isnothing(fuzzy)
            tau_T_cl = fact_deriv * beta_p[deriv + 1, 2]
            tau_T_bc = fact_deriv * beta_bc[deriv + 1, 2]
            tau_Y_cl = tau_cl
            tau_Y_bc = tau_bc
            
            tau_cl = tau_Y_cl / tau_T_cl
            s_Y = [1/tau_T_cl, -(tau_Y_cl / tau_T_cl^2)]
            B_F = [tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc]
            tau_bc = tau_cl - (s_Y' * B_F)
            
            tau_T_cl_l = fact_deriv * beta_p_l[deriv + 1, 2]
            tau_T_cl_r = fact_deriv * beta_p_r[deriv + 1, 2]
            tau_T_bc_l = fact_deriv * beta_bc_l[deriv + 1, 2]
            tau_T_bc_r = fact_deriv * beta_bc_r[deriv + 1, 2]
            
            B_F_l = [tau_Y_cl_l - tau_Y_bc_l, tau_T_cl_l - tau_T_bc_l]
            B_F_r = [tau_Y_cl_r - tau_Y_bc_r, tau_T_cl_r - tau_T_bc_r]
            bias_l = s_Y' * B_F_l
            bias_r = s_Y' * B_F_r
        end
    else
        # With covariates
        U_p_l = (R_p_l .* reshape(W_h_l, :, 1))' * D_l
        U_p_r = (R_p_r .* reshape(W_h_r, :, 1))' * D_r
        
        ZWD_p_l = (eZ_l .* reshape(W_h_l, :, 1))' * D_l
        ZWD_p_r = (eZ_r .* reshape(W_h_r, :, 1))' * D_r
        
        colsZ = (1 + dT + 1):size(D_l, 2)
        UiGU_p_l = U_p_l[:, colsZ]' * invG_p_l * U_p_l
        UiGU_p_r = U_p_r[:, colsZ]' * invG_p_r * U_p_r
        
        ZWZ_p = (ZWD_p_l[:, colsZ] .- UiGU_p_l[:, colsZ]) .+ (ZWD_p_r[:, colsZ] .- UiGU_p_r[:, colsZ])
        ZWY_p = (ZWD_p_l[:, 1:(1+dT)] .- UiGU_p_l[:, 1:(1+dT)]) .+ (ZWD_p_r[:, 1:(1+dT)] .- UiGU_p_r[:, 1:(1+dT)])
        
        gamma_p = pinv(ZWZ_p) * ZWY_p
        s_Y = vcat(1.0, -gamma_p[:, 1])
        
        if isnothing(fuzzy)
            tau_cl = scalepar * (s_Y' * beta_p[deriv + 1, :])
            tau_bc = scalepar * (s_Y' * beta_bc[deriv + 1, :])
            bias_l = scalepar * (s_Y' * (beta_p_l[deriv + 1, :] .- beta_bc_l[deriv + 1, :]))
            bias_r = scalepar * (s_Y' * (beta_p_r[deriv + 1, :] .- beta_bc_r[deriv + 1, :]))
        else
            s_T = vcat(1.0, -gamma_p[:, 2])
            
            idx_Y = vcat(1, collect(colsZ))
            idx_T = vcat(2, collect(colsZ))
            
            tau_Y_cl = scalepar * fact_deriv * (s_Y' * beta_p[deriv + 1, idx_Y])
            tau_T_cl = fact_deriv * (s_T' * beta_p[deriv + 1, idx_T])
            tau_Y_bc = scalepar * fact_deriv * (s_Y' * beta_bc[deriv + 1, idx_Y])
            tau_T_bc = fact_deriv * (s_T' * beta_bc[deriv + 1, idx_T])
            
            tau_cl = tau_Y_cl / tau_T_cl
            B_F = [tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc]
            s_Y_fuzzy = [1/tau_T_cl, -(tau_Y_cl / tau_T_cl^2)]
            tau_bc = tau_cl - (s_Y_fuzzy' * B_F)
            
            tau_Y_cl_l = scalepar * fact_deriv * (s_Y' * beta_p_l[deriv + 1, idx_Y])
            tau_Y_bc_l = scalepar * fact_deriv * (s_Y' * beta_bc_l[deriv + 1, idx_Y])
            tau_T_cl_l = fact_deriv * (s_T' * beta_p_l[deriv + 1, idx_T])
            tau_T_bc_l = fact_deriv * (s_T' * beta_bc_l[deriv + 1, idx_T])
            
            tau_Y_cl_r = scalepar * fact_deriv * (s_Y' * beta_p_r[deriv + 1, idx_Y])
            tau_Y_bc_r = scalepar * fact_deriv * (s_Y' * beta_bc_r[deriv + 1, idx_Y])
            tau_T_cl_r = fact_deriv * (s_T' * beta_p_r[deriv + 1, idx_T])
            tau_T_bc_r = fact_deriv * (s_T' * beta_bc_r[deriv + 1, idx_T])
            
            B_F_l = [tau_Y_cl_l - tau_Y_bc_l, tau_T_cl_l - tau_T_bc_l]
            B_F_r = [tau_Y_cl_r - tau_Y_bc_r, tau_T_cl_r - tau_T_bc_r]
            bias_l = s_Y_fuzzy' * B_F_l
            bias_r = s_Y_fuzzy' * B_F_r
            
            # Combine s_Y for VCE
            s_Y = vcat(s_Y_fuzzy, -(1/tau_T_cl) .* gamma_p[:, 1] .+ (tau_Y_cl / tau_T_cl^2) .* gamma_p[:, 2])
        end
    end
    
    # Variance estimation
    hii_l = hii_r = nothing
    predicts_p_l = R_p_l * beta_p_l
    predicts_p_r = R_p_r * beta_p_r
    predicts_q_l = R_q_l * beta_q_l
    predicts_q_r = R_q_r * beta_q_r
    
    if vce in ["hc2", "hc3"]
        hii_l = sum((R_p_l * invG_p_l) .* (R_p_l .* reshape(W_h_l, :, 1)), dims=2)
        hii_r = sum((R_p_r * invG_p_r) .* (R_p_r .* reshape(W_h_r, :, 1)), dims=2)
    end
    
    eC_l = isnothing(cluster) ? nothing : cluster[x .< c][ind_l]
    eC_r = isnothing(cluster) ? nothing : cluster[x .>= c][ind_r]
    
    res_h_l = rdrobust_res(eX_l, eY_l, isnothing(fuzzy) ? nothing : eT_l, isnothing(covs) ? nothing : eZ_l, predicts_p_l, hii_l, vce, nnmatch, dups_l, dupsid_l, p + 1)
    res_h_r = rdrobust_res(eX_r, eY_r, isnothing(fuzzy) ? nothing : eT_r, isnothing(covs) ? nothing : eZ_r, predicts_p_r, hii_r, vce, nnmatch, dups_r, dupsid_r, p + 1)
    
    res_b_l = vce == "nn" ? res_h_l : rdrobust_res(eX_l, eY_l, isnothing(fuzzy) ? nothing : eT_l, isnothing(covs) ? nothing : eZ_l, predicts_q_l, hii_l, vce, nnmatch, dups_l, dupsid_l, q + 1)
    res_b_r = vce == "nn" ? res_h_r : rdrobust_res(eX_r, eY_r, isnothing(fuzzy) ? nothing : eT_r, isnothing(covs) ? nothing : eZ_r, predicts_q_r, hii_r, vce, nnmatch, dups_r, dupsid_r, q + 1)
    
    V_Y_cl_l = invG_p_l * rdrobust_vce(dT + dZ, s_Y, R_p_l .* reshape(W_h_l, :, 1), res_h_l, eC_l) * invG_p_l
    V_Y_cl_r = invG_p_r * rdrobust_vce(dT + dZ, s_Y, R_p_r .* reshape(W_h_r, :, 1), res_h_r, eC_r) * invG_p_r
    V_Y_rb_l = invG_p_l * rdrobust_vce(dT + dZ, s_Y, Q_q_l, res_b_l, eC_l) * invG_p_l
    V_Y_rb_r = invG_p_r * rdrobust_vce(dT + dZ, s_Y, Q_q_r, res_b_r, eC_r) * invG_p_r
    
    V_tau_cl = scalepar^2 * fact_deriv^2 * (V_Y_cl_l[deriv + 1, deriv + 1] + V_Y_cl_r[deriv + 1, deriv + 1])
    V_tau_rb = scalepar^2 * fact_deriv^2 * (V_Y_rb_l[deriv + 1, deriv + 1] + V_Y_rb_r[deriv + 1, deriv + 1])
    
    se_tau_cl = sqrt(V_tau_cl)
    se_tau_rb = sqrt(V_tau_rb)
    
    tau = [tau_cl, tau_bc, tau_bc]
    se = [se_tau_cl, se_tau_cl, se_tau_rb]
    t_stat = tau ./ se
    quant = quantile(Normal(), 1 - (1 - level/100)/2)
    pv = 2 .* ccdf.(Normal(), abs.(t_stat))
    ci = hcat(tau .- quant .* se, tau .+ quant .* se)
    
    # Prepare outputs
    Estimate = DataFrame(tau_us=tau_cl, tau_bc=tau_bc, se_us=se_tau_cl, se_rb=se_tau_rb)
    bws = DataFrame(h_left=h_l, h_right=h_r, b_left=b_l, b_right=b_r)
    labels = ["Conventional", "Bias-Corrected", "Robust"]
    coef_df = DataFrame(Method=labels, Coeff=tau)
    se_df = DataFrame(Method=labels, StdErr=se)
    t_df = DataFrame(Method=labels, tStat=t_stat)
    pv_df = DataFrame(Method=labels, PValue=pv)
    ci_df = DataFrame(Method=labels, CILower=ci[:, 1], CIUpper=ci[:, 2])
    
    return RDRobustOutput(Estimate, bws, coef_df, se_df, t_df, pv_df, ci_df,
                          beta_p_l, beta_p_r, V_Y_cl_l, V_Y_cl_r, V_Y_rb_l, V_Y_rb_r,
                          [N_l, N_r], [N_h_l, N_h_r], [N_b_l, N_b_r], [M_l, M_r],
                          [tau_cl, tau_cl], [tau_bc, tau_bc], c, p, q, [bias_l, bias_r],
                          kernel, all_results, vce, bwselect, level, masspoints)
end

end # module
