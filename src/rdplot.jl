module RDPlotEstimation

using ..Utils
using Missings
using Statistics
using DataFrames
using LinearAlgebra
using Distributions

export rdplot, RDPlotOutput

struct RDPlotOutput
    coef::DataFrame
    vars_bins::DataFrame
    vars_poly::DataFrame
    J::Vector{Int}
    J_IMSE::Vector{Float64}
    J_MV::Vector{Float64}
    scale::Vector{Float64}
    rscale::Vector{Float64}
    bin_avg::Vector{Float64}
    bin_med::Vector{Float64}
    p::Int
    c::Float64
    h::Vector{Float64}
    N::Vector{Int}
    N_h::Vector{Int}
    binselect::String
    kernel::String
end

function rdplot(y, x; c=0.0, p=4, nbins=nothing, binselect="esmv", scale=nothing,
                kernel="uni", weights=nothing, h=nothing, 
                covs=nothing, covs_eval="mean", covs_drop=true,
                support=nothing, subset=nothing, masspoints="adjust",
                hide=false, ci=nothing, shade=false)

    # Input cleaning
    x = collect(Missings.replace(reshape(x, :), NaN))
    y = collect(Missings.replace(reshape(y, :), NaN))
    
    if !isnothing(subset)
        x = x[subset]
        y = y[subset]
        if !isnothing(weights) weights = collect(Missings.replace(weights[subset], NaN)) end
        if !isnothing(covs) covs = collect(Missings.replace(covs[subset, :], NaN)) end
    end
    
    na_ok = complete_cases(x) .& complete_cases(y)
    if !isnothing(covs) na_ok .&= complete_cases(covs) end
    if !isnothing(weights) na_ok .&= complete_cases(weights) .& (weights .>= 0) end
    
    x = Float64.(x[na_ok])
    y = Float64.(y[na_ok])
    if !isnothing(covs) covs = Float64.(covs[na_ok, :]) end
    if !isnothing(weights) weights = Float64.(weights[na_ok]) end
    
    x_min = minimum(x)
    x_max = maximum(x)
    
    if !isnothing(support)
        support_l, support_r = support
        x_min = min(x_min, support_l)
        x_max = max(x_max, support_r)
    end
    
    x_l = x[x .< c]
    x_r = x[x .>= c]
    y_l = y[x .< c]
    y_r = y[x .>= c]
    
    n_l = length(x_l)
    n_r = length(x_r)
    n = n_l + n_r
    
    scale_l = scale_r = 1.0
    if !isnothing(scale)
        if scale isa Real
            scale_l = scale_r = Float64(scale)
        else
            scale_l, scale_r = Float64.(scale)
        end
    end
    
    nbins_l = nbins_r = 0
    if !isnothing(nbins)
        if nbins isa Real
            nbins_l = nbins_r = Int(nbins)
        else
            nbins_l, nbins_r = Int.(nbins)
        end
    end

    h_l = h_r = 0.0
    if isnothing(h)
        h_l = c - x_min
        h_r = x_max - c
    else
        if h isa Real
            h_l = h_r = Float64(h)
        else
            h_l, h_r = Float64.(h)
        end
    end
    
    flag_no_ci = isnothing(ci)
    if isnothing(ci) ci = 95.0 end
    
    kernel_type = "Uniform"
    if kernel in ["epanechnikov", "epa"] kernel_type = "Epanechnikov" end
    if kernel in ["triangular", "tri"] kernel_type = "Triangular" end
    
    # Mass points
    binselect_type = binselect
    if masspoints in ["check", "adjust"]
        M_l = length(unique(x_l))
        M_r = length(unique(x_r))
        if (1 - M_l/n_l >= 0.2 || 1 - M_r/n_r >= 0.2)
            @info "Mass points detected in the running variable."
            if masspoints == "adjust"
                if binselect == "es" binselect_type = "espr" end
                if binselect == "esmv" binselect_type = "esmvpr" end
                if binselect == "qs" binselect_type = "qspr" end
                if binselect == "qsmv" binselect_type = "qsmvpr" end
            end
        end
    end
    
    # Collinearity
    dZ = 0
    if !isnothing(covs)
        if covs_drop
            covs = covs_drop_fun(covs)
        end
        dZ = size(covs, 2)
    end
    
    # Global polynomial fit
    R_p_l = fill(NaN, n_l, p + 1)
    R_p_r = fill(NaN, n_r, p + 1)
    for j in 0:p
        R_p_l[:, j + 1] = (x_l .- c).^j
        R_p_r[:, j + 1] = (x_r .- c).^j
    end
    
    W_h_l = rdrobust_kweight(x_l, c, h_l, kernel)
    W_h_r = rdrobust_kweight(x_r, c, h_r, kernel)
    n_h_l = sum(W_h_l .> 0)
    n_h_r = sum(W_h_r .> 0)
    
    if !isnothing(weights)
        W_h_l .*= weights[x .< c]
        W_h_r .*= weights[x .>= c]
    end
    
    invG_p_l = qrXXinv(R_p_l .* sqrt.(reshape(W_h_l, :, 1)))
    invG_p_r = qrXXinv(R_p_r .* sqrt.(reshape(W_h_r, :, 1)))
    
    gamma_p1_l = zeros(p + 1)
    gamma_p1_r = zeros(p + 1)
    gamma_p = nothing
    
    if isnothing(covs)
        gamma_p1_l = invG_p_l * ((R_p_l .* reshape(W_h_l, :, 1))' * y_l)
        gamma_p1_r = invG_p_r * ((R_p_r .* reshape(W_h_r, :, 1))' * y_r)
    else
        z_l = covs[x .< c, :]
        z_r = covs[x .>= c, :]
        D_l = hcat(reshape(y_l, :, 1), z_l)
        D_r = hcat(reshape(y_r, :, 1), z_r)
        
        U_p_l = (R_p_l .* reshape(W_h_l, :, 1))' * D_l
        U_p_r = (R_p_r .* reshape(W_h_r, :, 1))' * D_r
        
        beta_p_l = invG_p_l * U_p_l
        beta_p_r = invG_p_r * U_p_r
        
        ZWD_p_l = (z_l .* reshape(W_h_l, :, 1))' * D_l
        ZWD_p_r = (z_r .* reshape(W_h_r, :, 1))' * D_r
        
        colsZ = 2:size(D_l, 2)
        UiGU_p_l = U_p_l[:, colsZ]' * invG_p_l * U_p_l
        UiGU_p_r = U_p_r[:, colsZ]' * invG_p_r * U_p_r
        
        ZWZ_p = (ZWD_p_l[:, colsZ] .- UiGU_p_l[:, colsZ]) .+ (ZWD_p_r[:, colsZ] .- UiGU_p_r[:, colsZ])
        ZWY_p = (ZWD_p_l[:, 1] .- UiGU_p_l[:, 1]) .+ (ZWD_p_r[:, 1] .- UiGU_p_r[:, 1])
        
        gamma_p = pinv(ZWZ_p) * ZWY_p
        s_Y = vcat(1.0, -gamma_p)
        
        gamma_p1_l = (s_Y' * beta_p_l')'
        gamma_p1_r = (s_Y' * beta_p_r')'
    end
    
    # Polynomial curve data
    nplot = 500
    x_plot_l = collect(range(c - h_l, c, length=nplot))
    x_plot_r = collect(range(c, c + h_r, length=nplot))
    rplot_l = fill(NaN, nplot, p + 1)
    rplot_r = fill(NaN, nplot, p + 1)
    for j in 0:p
        rplot_l[:, j + 1] = (x_plot_l .- c).^j
        rplot_r[:, j + 1] = (x_plot_r .- c).^j
    end
    y_hat_l = rplot_l * gamma_p1_l
    y_hat_r = rplot_r * gamma_p1_r
    
    if !isnothing(covs) && covs_eval == "mean"
        gammaZ = mean(covs, dims=1) * gamma_p
        y_hat_l .+= gammaZ[1]
        y_hat_r .+= gammaZ[1]
    end
    
    # Optimal Bins
    invG_k_l = invG_k_r = nothing
    rk_l = rk_r = nothing
    k_selected = 0
    for k in 4:-1:2
        try
            tk_l = fill(NaN, n_l, k + 1)
            tk_r = fill(NaN, n_r, k + 1)
            for i in 0:k
                tk_l[:, i + 1] = x_l.^i
                tk_r[:, i + 1] = x_r.^i
            end
            invG_k_l = qrXXinv(tk_l)
            invG_k_r = qrXXinv(tk_r)
            rk_l = tk_l
            rk_r = tk_r
            k_selected = k
            break
        catch
        end
    end
    
    gamma_k1_l = invG_k_l * (rk_l' * y_l)
    gamma_k2_l = invG_k_l * (rk_l' * (y_l.^2))
    gamma_k1_r = invG_k_r * (rk_r' * y_r)
    gamma_k2_r = invG_k_r * (rk_r' * (y_r.^2))
    
    drk_l = fill(NaN, n_l, k_selected)
    drk_r = fill(NaN, n_r, k_selected)
    for j in 1:k_selected
        drk_l[:, j] = j .* x_l.^(j-1)
        drk_r[:, j] = j .* x_r.^(j-1)
    end
    
    ind_l = sortperm(x_l)
    ind_r = sortperm(x_r)
    x_i_l, y_i_l = x_l[ind_l], y_l[ind_l]
    x_i_r, y_i_r = x_r[ind_r], y_r[ind_r]
    
    dxi_l = diff(x_i_l)
    dxi_r = diff(x_i_r)
    dyi_l = diff(y_i_l)
    dyi_r = diff(y_i_r)
    
    x_bar_i_l = (x_i_l[1:end-1] .+ x_i_l[2:end]) ./ 2
    x_bar_i_r = (x_i_r[1:end-1] .+ x_i_r[2:end]) ./ 2
    
    rk_i_l = fill(NaN, n_l - 1, k_selected + 1)
    rk_i_r = fill(NaN, n_r - 1, k_selected + 1)
    drk_i_l = fill(NaN, n_l - 1, k_selected)
    drk_i_r = fill(NaN, n_r - 1, k_selected)
    for j in 0:k_selected
        rk_i_l[:, j + 1] = x_bar_i_l.^j
        rk_i_r[:, j + 1] = x_bar_i_r.^j
        if j > 0
            drk_i_l[:, j] = j .* x_bar_i_l.^(j-1)
            drk_i_r[:, j] = j .* x_bar_i_r.^(j-1)
        end
    end
    
    mu1_i_hat_l = drk_i_l * gamma_k1_l[2:end]
    mu1_i_hat_r = drk_i_r * gamma_k1_r[2:end]
    
    mu0_i_hat_l = rk_i_l * gamma_k1_l
    mu0_i_hat_r = rk_i_r * gamma_k1_r
    mu2_i_hat_l = rk_i_l * gamma_k2_l
    mu2_i_hat_r = rk_i_r * gamma_k2_r
    
    mu0_hat_l = rk_l * gamma_k1_l
    mu0_hat_r = rk_r * gamma_k1_r
    mu2_hat_l = rk_l * gamma_k2_l
    mu2_hat_r = rk_r * gamma_k2_r
    
    mu1_hat_l = drk_l * gamma_k1_l[2:end]
    mu1_hat_r = drk_r * gamma_k1_r[2:end]
    
    var_y_l = var(y_l)
    var_y_r = var(y_r)
    
    sigma2_hat_l_bar = max.(0.0, mu2_i_hat_l .- mu0_i_hat_l.^2)
    sigma2_hat_r_bar = max.(0.0, mu2_i_hat_r .- mu0_i_hat_r.^2)
    sigma2_hat_l = max.(0.0, mu2_hat_l .- mu0_hat_l.^2)
    sigma2_hat_r = max.(0.0, mu2_hat_r .- mu0_hat_r.^2)
    
    J_fun(B, V) = ceil.((((2 .* B) ./ V) .* n).^(1/3))
    
    B_es_hat_dw = [((c - x_min)^2 / (12 * n)) * sum(mu1_hat_l.^2), ((x_max - c)^2 / (12 * n)) * sum(mu1_hat_r.^2)]
    V_es_hat_dw = [(0.5 / (c - x_min)) * sum(dxi_l .* dyi_l.^2), (0.5 / (x_max - c)) * sum(dxi_r .* dyi_r.^2)]
    V_es_chk_dw = [(1.0 / (c - x_min)) * sum(dxi_l .* sigma2_hat_l_bar), (1.0 / (x_max - c)) * sum(dxi_r .* sigma2_hat_r_bar)]
    
    J_es_hat_dw = J_fun(B_es_hat_dw, V_es_hat_dw)
    J_es_chk_dw = J_fun(B_es_hat_dw, V_es_chk_dw)
    
    B_qs_hat_dw = [(n_l^2 / (24 * n)) * sum(dxi_l.^2 .* mu1_i_hat_l.^2), (n_r^2 / (24 * n)) * sum(dxi_r.^2 .* mu1_i_hat_r.^2)]
    V_qs_hat_dw = [(1.0 / (2 * n_l)) * sum(dyi_l.^2), (1.0 / (2 * n_r)) * sum(dyi_r.^2)]
    V_qs_chk_dw = [(1.0 / n_l) * sum(sigma2_hat_l), (1.0 / n_r) * sum(sigma2_hat_r)]
    
    J_qs_hat_dw = J_fun(B_qs_hat_dw, V_qs_hat_dw)
    J_qs_chk_dw = J_fun(B_qs_hat_dw, V_qs_chk_dw)
    
    J_es_hat_mv = [ceil((var_y_l / V_es_hat_dw[1]) * (n / log(n)^2)), ceil((var_y_r / V_es_hat_dw[2]) * (n / log(n)^2))]
    J_es_chk_mv = [ceil((var_y_l / V_es_chk_dw[1]) * (n / log(n)^2)), ceil((var_y_r / V_es_chk_dw[2]) * (n / log(n)^2))]
    J_qs_hat_mv = [ceil((var_y_l / V_qs_hat_dw[1]) * (n / log(n)^2)), ceil((var_y_r / V_qs_hat_dw[2]) * (n / log(n)^2))]
    J_qs_chk_mv = [ceil((var_y_l / V_qs_chk_dw[1]) * (n / log(n)^2)), ceil((var_y_r / V_qs_chk_dw[2]) * (n / log(n)^2))]
    
    J_star_orig = J_es_hat_mv
    meth = "es"
    J_IMSE = J_es_hat_dw
    J_MV = J_es_hat_mv
    
    if binselect_type == "es"
        J_star_orig = J_es_hat_dw
        meth = "es"; J_IMSE = J_es_hat_dw; J_MV = J_es_hat_mv
    elseif binselect_type == "espr"
        J_star_orig = J_es_chk_dw
        meth = "es"; J_IMSE = J_es_chk_dw; J_MV = J_es_chk_mv
    elseif binselect_type == "esmv"
        J_star_orig = J_es_hat_mv
        meth = "es"; J_IMSE = J_es_hat_dw; J_MV = J_es_hat_mv
    elseif binselect_type == "esmvpr"
        J_star_orig = J_es_chk_mv
        meth = "es"; J_IMSE = J_es_chk_dw; J_MV = J_es_chk_mv
    elseif binselect_type == "qs"
        J_star_orig = J_qs_hat_dw
        meth = "qs"; J_IMSE = J_qs_hat_dw; J_MV = J_qs_hat_mv
    elseif binselect_type == "qspr"
        J_star_orig = J_qs_chk_dw
        meth = "qs"; J_IMSE = J_qs_chk_dw; J_MV = J_qs_chk_mv
    elseif binselect_type == "qsmv"
        J_star_orig = J_qs_hat_mv
        meth = "qs"; J_IMSE = J_qs_hat_dw; J_MV = J_qs_hat_mv
    elseif binselect_type == "qsmvpr"
        J_star_orig = J_qs_chk_mv
        meth = "qs"; J_IMSE = J_qs_chk_dw; J_MV = J_qs_chk_mv
    end
    
    J_star_l = Int(scale_l * J_star_orig[1])
    J_star_r = Int(scale_r * J_star_orig[2])
    
    if !isnothing(nbins)
        J_star_l = nbins_l
        J_star_r = nbins_r
    end
    
    if var_y_l == 0.0 J_star_l = 1 end
    if var_y_r == 0.0 J_star_r = 1 end
    
    rscale_l = J_star_l / J_IMSE[1]
    rscale_r = J_star_r / J_IMSE[2]
    
    jumps_l = meth == "es" ? collect(range(x_min, c, length=J_star_l + 1)) : quantile(x_l, range(0, 1, length=J_star_l + 1))
    jumps_r = meth == "es" ? collect(range(c, x_max, length=J_star_r + 1)) : quantile(x_r, range(0, 1, length=J_star_r + 1))
    
    function find_interval(val, jumps, side)
        idx = searchsortedfirst(jumps, val)
        if side == "right"
            idx = searchsortedlast(jumps, val)
        end
        return idx
    end
    
    bin_x_l = [find_interval(v, jumps_l, "right") for v in x_l]
    bin_x_l = clamp.(bin_x_l, 1, J_star_l)
    bin_x_r = [find_interval(v, jumps_r, "left") for v in x_r]
    bin_x_r = clamp.(bin_x_r, 1, J_star_r)
    
    # Bin means
    df_l = DataFrame(bin=bin_x_l, y=y_l, x=x_l)
    rdplot_l = combine(groupby(df_l, :bin), :y => mean => :y_mean, :x => mean => :x_mean, :y => length => :n, :y => std => :y_std)
    
    df_r = DataFrame(bin=bin_x_r, y=y_r, x=x_r)
    rdplot_r = combine(groupby(df_r, :bin), :y => mean => :y_mean, :x => mean => :x_mean, :y => length => :n, :y => std => :y_std)
    
    # Fill missing bins
    full_l = DataFrame(bin=1:J_star_l)
    rdplot_l = leftjoin(full_l, rdplot_l, on=:bin)
    full_r = DataFrame(bin=1:J_star_r)
    rdplot_r = leftjoin(full_r, rdplot_r, on=:bin)
    
    rdplot_mean_bin_l = [(jumps_l[i] + jumps_l[i+1])/2 for i in 1:J_star_l]
    rdplot_mean_bin_r = [(jumps_r[i] + jumps_r[i+1])/2 for i in 1:J_star_r]
    
    # Handle missings in rdplot_l/r
    rdplot_l.n = coalesce.(rdplot_l.n, 0)
    rdplot_r.n = coalesce.(rdplot_r.n, 0)
    rdplot_l.y_mean = coalesce.(rdplot_l.y_mean, NaN)
    rdplot_r.y_mean = coalesce.(rdplot_r.y_mean, NaN)
    rdplot_l.x_mean = coalesce.(rdplot_l.x_mean, NaN)
    rdplot_r.x_mean = coalesce.(rdplot_r.x_mean, NaN)
    rdplot_l.y_std = coalesce.(rdplot_l.y_std, 0.0)
    rdplot_r.y_std = coalesce.(rdplot_r.y_std, 0.0)

    vars_bins = DataFrame(
        rdplot_mean_bin = vcat(rdplot_mean_bin_l, rdplot_mean_bin_r),
        rdplot_mean_x = vcat(rdplot_l.x_mean, rdplot_r.x_mean),
        rdplot_mean_y = vcat(rdplot_l.y_mean, rdplot_r.y_mean),
        rdplot_min_bin = vcat(jumps_l[1:end-1], jumps_r[1:end-1]),
        rdplot_max_bin = vcat(jumps_l[2:end], jumps_r[2:end]),
        rdplot_se_y = vcat(rdplot_l.y_std ./ sqrt.(max.(1, rdplot_l.n)), rdplot_r.y_std ./ sqrt.(max.(1, rdplot_r.n))),
        rdplot_N = vcat(rdplot_l.n, rdplot_r.n)
    )
    
    # Confidence intervals
    t_dist = TDist.(max.(1, vars_bins.rdplot_N .- 1))
    quant = .- quantile.(t_dist, (1 - ci/100)/2)
    vars_bins.rdplot_ci_l = vars_bins.rdplot_mean_y .- quant .* vars_bins.rdplot_se_y
    vars_bins.rdplot_ci_r = vars_bins.rdplot_mean_y .+ quant .* vars_bins.rdplot_se_y
    
    vars_poly = DataFrame(
        rdplot_x = vcat(x_plot_l, x_plot_r),
        rdplot_y = vcat(y_hat_l, y_hat_r)
    )
    
    coef_df = DataFrame(Left=gamma_p1_l, Right=gamma_p1_r)
    
    bin_avg = [mean(diff(jumps_l)), mean(diff(jumps_r))]
    bin_med = [median(diff(jumps_l)), median(diff(jumps_r))]
    
    return RDPlotOutput(coef_df, vars_bins, vars_poly, [J_star_l, J_star_r], J_IMSE, J_MV, [scale_l, scale_r], [rscale_l, rscale_r], bin_avg, bin_med, p, c, [h_l, h_r], [n_l, n_r], [n_h_l, n_h_r], binselect_type, kernel_type)
end

end # module
