module BandwidthSelection

using ..Utils
using Missings
using Statistics
using DataFrames
using LinearAlgebra
using Distributions

export rdbwselect, RDBWSelectOutput

struct RDBWSelectOutput
    bws::DataFrame
    bwselect::String
    bw_list::Vector{String}
    kernel::String
    p::Int
    q::Int
    c::Float64
    N::Vector{Int}
    M::Vector{Int}
    vce::String
    masspoints::String
end

function rdbwselect(y, x; c=0.0, fuzzy=nothing, deriv=0, p=nothing, q=nothing,
                   covs=nothing, covs_drop=true, kernel="tri", weights=nothing, 
                   bwselect="mserd", vce="nn", cluster=nothing, nnmatch=3,
                   scaleregul=1.0, sharpbw=false, all_bws=false, subset=nothing,
                   masspoints="adjust", bwcheck=nothing, bwrestrict=true,
                   stdvars=false)

    # Input cleaning
    # Convert to Float64 and handle Missing
    x = collect(Missings.replace(reshape(x, :), NaN))
    y = collect(Missings.replace(reshape(y, :), NaN))
    
    if !isnothing(subset)
        x = x[subset]
        y = y[subset]
        if !isnothing(fuzzy) fuzzy = collect(Missings.replace(fuzzy[subset], NaN)) end
        if !isnothing(cluster) cluster = collect(Missings.replace(cluster[subset], NaN)) end
        if !isnothing(weights) weights = collect(Missings.replace(weights[subset], NaN)) end
        if !isnothing(covs) covs = collect(Missings.replace(covs[subset, :], NaN)) end
    end
    
    x = Float64.(x)
    y = Float64.(y)
    
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
    
    # Standardize vars
    x_iq = quantile(x, 0.75) - quantile(x, 0.25)
    BWp = min(std(x), x_iq / 1.349)
    x_sd = 1.0
    y_sd = 1.0
    if stdvars
        y_sd = std(y)
        x_sd = std(x)
        y ./= y_sd
        x ./= x_sd
        c /= x_sd
        BWp = min(1.0, (x_iq / x_sd) / 1.349)
    end
    
    X_l = x[x .< c]
    X_r = x[x .>= c]
    Y_l = y[x .< c]
    Y_r = y[x .>= c]
    
    N_l = length(X_l)
    N_r = length(X_r)
    N = N_l + N_r
    
    M_l = N_l
    M_r = N_r
    X_uniq_l = Float64[]
    X_uniq_r = Float64[]
    
    if masspoints in ["check", "adjust"]
        X_uniq_l = sort(unique(X_l), rev=true)
        X_uniq_r = sort(unique(X_r))
        M_l = length(X_uniq_l)
        M_r = length(X_uniq_r)
        if (1 - M_l/N_l >= 0.1 || 1 - M_r/N_r >= 0.1)
            @info "Mass points detected in the running variable."
            if masspoints == "adjust" && isnothing(bwcheck)
                bwcheck = 10
            end
        end
    end
    
    covs_drop_coll = covs_drop ? 1 : 0
    if !isnothing(covs)
        if covs_drop
            covs = covs_drop_fun(covs)
        end
    end
    
    kernel_type = ""
    C_c = 0.0
    if kernel in ["epanechnikov", "epa"]
        kernel_type = "Epanechnikov"
        C_c = 2.34
    elseif kernel in ["uniform", "uni"]
        kernel_type = "Uniform"
        C_c = 1.843
    else
        kernel_type = "Triangular"
        C_c = 2.576
    end
    
    dups_l = zeros(Int, N_l)
    dupsid_l = zeros(Int, N_l)
    dups_r = zeros(Int, N_r)
    dupsid_r = zeros(Int, N_r)
    
    if vce == "nn"
        # Left side
        for i in 1:N_l
            d = count(==(X_l[i]), X_l)
            dups_l[i] = d
            # find first occurrence of X_l[i] in X_l
            first_idx = findfirst(==(X_l[i]), X_l)
            dupsid_l[i] = i - first_idx + 1
        end
        # Right side
        for i in 1:N_r
            d = count(==(X_r[i]), X_r)
            dups_r[i] = d
            first_idx = findfirst(==(X_r[i]), X_r)
            dupsid_r[i] = i - first_idx + 1
        end
    end
    
    Z_l = isnothing(covs) ? nothing : covs[x .< c, :]
    Z_r = isnothing(covs) ? nothing : covs[x .>= c, :]
    
    T_l = nothing
    T_r = nothing
    if !isnothing(fuzzy)
        T_l = fuzzy[x .< c]
        T_r = fuzzy[x .>= c]
        if var(T_l) == 0 || var(T_r) == 0 || sharpbw
            T_l = nothing
            T_r = nothing
        end
    end
    
    C_l = isnothing(cluster) ? nothing : cluster[x .< c]
    C_r = isnothing(cluster) ? nothing : cluster[x .>= c]
    g_l = isnothing(C_l) ? 0 : length(unique(C_l))
    g_r = isnothing(C_r) ? 0 : length(unique(C_r))
    
    fw_l = isnothing(weights) ? nothing : weights[x .< c]
    fw_r = isnothing(weights) ? nothing : weights[x .>= c]
    
    c_bw = C_c * BWp * (masspoints == "adjust" ? (M_l + M_r) : N)^(-1/5)
    
    if bwrestrict
        bw_max_l = abs(c - minimum(x))
        bw_max_r = abs(c - maximum(x))
        c_bw = min(c_bw, max(bw_max_l, bw_max_r))
    end
    
    bw_min_l = 0.0
    bw_min_r = 0.0
    if !isnothing(bwcheck)
        bwcheck_l = min(bwcheck, M_l)
        bwcheck_r = min(bwcheck, M_r)
        bw_min_l = abs(X_uniq_l[bwcheck_l] - c) + 1e-8
        bw_min_r = abs(X_uniq_r[bwcheck_r] - c) + 1e-8
        c_bw = max(c_bw, bw_min_l, bw_min_r)
    end
    
    # Step 1: d_bw
    range_l = abs(c - minimum(X_l))
    range_r = abs(c - maximum(X_r))
    
    C_d_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, q+1, q+1, q+2, c_bw, range_l, 0, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
    C_d_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, q+1, q+1, q+2, c_bw, range_r, 0, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
    
    # Initialize BW variables
    h_mserd = h_msetwo_l = h_msetwo_r = h_msesum = 0.0
    b_mserd = b_msetwo_l = b_msetwo_r = b_msesum = 0.0
    
    # TWO bw
    if bwselect in ["msetwo", "certwo", "msecomb2", "cercomb2"] || all_bws
        d_bw_l = (C_d_l[1] / C_d_l[2]^2)^C_d_l[4]
        d_bw_r = (C_d_r[1] / C_d_r[2]^2)^C_d_l[4]
        if bwrestrict
            d_bw_l = min(d_bw_l, abs(c - minimum(X_l)))
            d_bw_r = min(d_bw_r, abs(c - maximum(X_r)))
        end
        if !isnothing(bwcheck)
            d_bw_l = max(d_bw_l, bw_min_l)
            d_bw_r = max(d_bw_r, bw_min_r)
        end
        
        C_b_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, q, p+1, q+1, c_bw, d_bw_l, scaleregul, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
        C_b_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, q, p+1, q+1, c_bw, d_bw_r, scaleregul, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
        
        b_bw_l = (C_b_l[1] / (C_b_l[2]^2 + scaleregul * C_b_l[3]))^C_b_l[4]
        b_bw_r = (C_b_r[1] / (C_b_r[2]^2 + scaleregul * C_b_r[3]))^C_b_l[4]
        if bwrestrict
            b_bw_l = min(b_bw_l, abs(c - minimum(X_l)))
            b_bw_r = min(b_bw_r, abs(c - maximum(X_r)))
        end
        
        C_h_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, p, deriv, q, c_bw, b_bw_l, scaleregul, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
        C_h_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, p, deriv, q, c_bw, b_bw_r, scaleregul, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
        
        h_bw_l = (C_h_l[1] / (C_h_l[2]^2 + scaleregul * C_h_l[3]))^C_h_l[4]
        h_bw_r = (C_h_r[1] / (C_h_r[2]^2 + scaleregul * C_h_r[3]))^C_h_l[4]
        if bwrestrict
            h_bw_l = min(h_bw_l, abs(c - minimum(X_l)))
            h_bw_r = min(h_bw_r, abs(c - maximum(X_r)))
        end
        h_msetwo_l = x_sd * h_bw_l
        h_msetwo_r = x_sd * h_bw_r
        b_msetwo_l = x_sd * b_bw_l
        b_msetwo_r = x_sd * b_bw_r
    end

    # SUM
    if bwselect in ["msesum", "cersum", "msecomb1", "msecomb2", "cercomb1", "cercomb2"] || all_bws
        d_bw_s = ((C_d_l[1] + C_d_r[1]) / (C_d_r[2] + C_d_l[2])^2)^C_d_l[4]
        if bwrestrict d_bw_s = min(d_bw_s, max(abs(c - minimum(x)), abs(c - maximum(x)))) end
        if !isnothing(bwcheck) d_bw_s = max(d_bw_s, bw_min_l, bw_min_r) end
        
        C_b_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, q, p+1, q+1, c_bw, d_bw_s, scaleregul, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
        C_b_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, q, p+1, q+1, c_bw, d_bw_s, scaleregul, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
        
        b_bw_s = ((C_b_l[1] + C_b_r[1]) / ((C_b_r[2] + C_b_l[2])^2 + scaleregul * (C_b_r[3] + C_b_l[3])))^C_b_l[4]
        if bwrestrict b_bw_s = min(b_bw_s, max(abs(c - minimum(x)), abs(c - maximum(x)))) end
        
        C_h_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, p, deriv, q, c_bw, b_bw_s, scaleregul, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
        C_h_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, p, deriv, q, c_bw, b_bw_s, scaleregul, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
        
        h_bw_s = ((C_h_l[1] + C_h_r[1]) / ((C_h_r[2] + C_h_l[2])^2 + scaleregul * (C_h_r[3] + C_h_l[3])))^C_h_l[4]
        if bwrestrict h_bw_s = min(h_bw_s, max(abs(c - minimum(x)), abs(c - maximum(x)))) end
        h_msesum = x_sd * h_bw_s
        b_msesum = x_sd * b_bw_s
    end

    # RD
    if bwselect in ["mserd", "cerrd", "msecomb1", "msecomb2", "cercomb1", "cercomb2", ""] || all_bws
        d_bw_d = ((C_d_l[1] + C_d_r[1]) / (C_d_r[2] - C_d_l[2])^2)^C_d_l[4]
        if bwrestrict d_bw_d = min(d_bw_d, max(abs(c - minimum(x)), abs(c - maximum(x)))) end
        if !isnothing(bwcheck) d_bw_d = max(d_bw_d, bw_min_l, bw_min_r) end
        
        C_b_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, q, p+1, q+1, c_bw, d_bw_d, scaleregul, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
        C_b_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, q, p+1, q+1, c_bw, d_bw_d, scaleregul, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
        
        b_bw_d = ((C_b_l[1] + C_b_r[1]) / ((C_b_r[2] - C_b_l[2])^2 + scaleregul * (C_b_r[3] + C_b_l[3])))^C_b_l[4]
        if bwrestrict b_bw_d = min(b_bw_d, max(abs(c - minimum(x)), abs(c - maximum(x)))) end
        
        C_h_l = rdrobust_bw(Y_l, X_l, T_l, Z_l, C_l, fw_l, c, p, deriv, q, c_bw, b_bw_d, scaleregul, vce, nnmatch, kernel, dups_l, dupsid_l, covs_drop_coll)
        C_h_r = rdrobust_bw(Y_r, X_r, T_r, Z_r, C_r, fw_r, c, p, deriv, q, c_bw, b_bw_d, scaleregul, vce, nnmatch, kernel, dups_r, dupsid_r, covs_drop_coll)
        
        h_bw_d = ((C_h_l[1] + C_h_r[1]) / ((C_h_r[2] - C_h_l[2])^2 + scaleregul * (C_h_r[3] + C_h_l[3])))^C_h_l[4]
        if bwrestrict h_bw_d = min(h_bw_d, max(abs(c - minimum(x)), abs(c - maximum(x)))) end
        h_mserd = x_sd * h_bw_d
        b_mserd = x_sd * b_bw_d
    end

    cer_h = (isnothing(cluster) ? N : (g_l + g_r))^(-(p / ((3+p) * (3+2*p))))
    cer_b = 1.0
    
    # Prepare bws matrix
    bw_list = String[]
    bws_data = Matrix{Float64}(undef, 0, 4)
    
    function add_bw(name, hl, hr, bl, br)
        push!(bw_list, name)
        bws_data = vcat(bws_data, [hl hr bl br])
        return bws_data
    end

    if all_bws
        bws_data = add_bw("mserd", h_mserd, h_mserd, b_mserd, b_mserd)
        bws_data = add_bw("msetwo", h_msetwo_l, h_msetwo_r, b_msetwo_l, b_msetwo_r)
        bws_data = add_bw("msesum", h_msesum, h_msesum, b_msesum, b_msesum)
        
        h_msecomb1 = min(h_mserd, h_msesum)
        b_msecomb1 = min(b_mserd, b_msesum)
        bws_data = add_bw("msecomb1", h_msecomb1, h_msecomb1, b_msecomb1, b_msecomb1)
        
        h_msecomb2_l = median([h_mserd, h_msesum, h_msetwo_l])
        h_msecomb2_r = median([h_mserd, h_msesum, h_msetwo_r])
        b_msecomb2_l = median([b_mserd, b_msesum, b_msetwo_l])
        b_msecomb2_r = median([b_mserd, b_msesum, b_msetwo_r])
        bws_data = add_bw("msecomb2", h_msecomb2_l, h_msecomb2_r, b_msecomb2_l, b_msecomb2_r)
        
        bws_data = add_bw("cerrd", h_mserd * cer_h, h_mserd * cer_h, b_mserd * cer_b, b_mserd * cer_b)
        bws_data = add_bw("certwo", h_msetwo_l * cer_h, h_msetwo_r * cer_h, b_msetwo_l * cer_b, b_msetwo_r * cer_b)
        bws_data = add_bw("cersum", h_msesum * cer_h, h_msesum * cer_h, b_msesum * cer_b, b_msesum * cer_b)
        bws_data = add_bw("cercomb1", h_msecomb1 * cer_h, h_msecomb1 * cer_h, b_msecomb1 * cer_b, b_msecomb1 * cer_b)
        bws_data = add_bw("cercomb2", h_msecomb2_l * cer_h, h_msecomb2_r * cer_h, b_msecomb2_l * cer_b, b_msecomb2_r * cer_b)
    else
        if bwselect == "mserd" || bwselect == ""
            bws_data = add_bw(bwselect == "" ? "mserd" : bwselect, h_mserd, h_mserd, b_mserd, b_mserd)
        elseif bwselect == "msetwo"
            bws_data = add_bw("msetwo", h_msetwo_l, h_msetwo_r, b_msetwo_l, b_msetwo_r)
        elseif bwselect == "msesum"
            bws_data = add_bw("msesum", h_msesum, h_msesum, b_msesum, b_msesum)
        elseif bwselect == "msecomb1"
            h_msecomb1 = min(h_mserd, h_msesum)
            b_msecomb1 = min(b_mserd, b_msesum)
            bws_data = add_bw("msecomb1", h_msecomb1, h_msecomb1, b_msecomb1, b_msecomb1)
        elseif bwselect == "msecomb2"
            h_msecomb2_l = median([h_mserd, h_msesum, h_msetwo_l])
            h_msecomb2_r = median([h_mserd, h_msesum, h_msetwo_r])
            b_msecomb2_l = median([b_mserd, b_msesum, b_msetwo_l])
            b_msecomb2_r = median([b_mserd, b_msesum, b_msetwo_r])
            bws_data = add_bw("msecomb2", h_msecomb2_l, h_msecomb2_r, b_msecomb2_l, b_msecomb2_r)
        elseif bwselect == "cerrd"
            bws_data = add_bw("cerrd", h_mserd * cer_h, h_mserd * cer_h, b_mserd * cer_b, b_mserd * cer_b)
        elseif bwselect == "certwo"
            bws_data = add_bw("certwo", h_msetwo_l * cer_h, h_msetwo_r * cer_h, b_msetwo_l * cer_b, b_msetwo_r * cer_b)
        elseif bwselect == "cersum"
            bws_data = add_bw("cersum", h_msesum * cer_h, h_msesum * cer_h, b_msesum * cer_b, b_msesum * cer_b)
        elseif bwselect == "cercomb1"
            h_msecomb1 = min(h_mserd, h_msesum)
            b_msecomb1 = min(b_mserd, b_msesum)
            bws_data = add_bw("cercomb1", h_msecomb1 * cer_h, h_msecomb1 * cer_h, b_msecomb1 * cer_b, b_msecomb1 * cer_b)
        elseif bwselect == "cercomb2"
            h_msecomb2_l = median([h_mserd, h_msesum, h_msetwo_l])
            h_msecomb2_r = median([h_mserd, h_msesum, h_msetwo_r])
            b_msecomb2_l = median([b_mserd, b_msesum, b_msetwo_l])
            b_msecomb2_r = median([b_mserd, b_msesum, b_msetwo_r])
            bws_data = add_bw("cercomb2", h_msecomb2_l * cer_h, h_msecomb2_r * cer_h, b_msecomb2_l * cer_b, b_msecomb2_r * cer_b)
        end
    end
    
    bws_df = DataFrame(bws_data, [:h_left, :h_right, :b_left, :b_right])
    
    return RDBWSelectOutput(bws_df, all_bws ? "All" : bwselect, bw_list, kernel_type, p, q, c, [N_l, N_r], [M_l, M_r], vce, masspoints)
end

end # module
