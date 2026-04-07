module Utils

using LinearAlgebra
using Statistics
using Distributions
using DataFrames

export rdrobust_kweight, rdrobust_res, rdrobust_vce, rdrobust_bw
export qrXXinv, crossprod, complete_cases, covs_drop_fun

function crossprod(x, y=nothing)
    if isnothing(y)
        return x' * x
    else
        return x' * y
    end
end

function qrXXinv(x)
    # Using inv(x'x) as in Python, but let's be slightly more robust
    return inv(x' * x)
end

function complete_cases(x::AbstractMatrix)
    return [!any(isnan, x[i, :]) for i in 1:size(x, 1)]
end

function complete_cases(x::AbstractVector)
    return .!isnan.(x)
end

function covs_drop_fun(z::AbstractMatrix, tol=1e-5)
    F = qr(z, Val(true))
    # F.R is the R matrix, F.p is the permutation vector
    r = F.R
    keep_idx = []
    for i in 1:min(size(r)...)
        if abs(r[i, i]) > tol
            push!(keep_idx, i)
        end
    end
    return z[:, F.p[keep_idx]]
end

function rdrobust_kweight(X, c, h, kernel)
    u = (X .- c) ./ h
    if kernel == "epanechnikov" || kernel == "epa"
        w = (0.75 .* (1 .- u.^2) .* (abs.(u) .<= 1)) ./ h
    elseif kernel == "uniform" || kernel == "uni"
        w = (0.5 .* (abs.(u) .<= 1)) ./ h
    else # triangular
        w = ((1 .- abs.(u)) .* (abs.(u) .<= 1)) ./ h
    end
    return w
end

function rdrobust_res(X, y, T, Z, m, hii, vce, matches, dups, dupsid, d)
    n = length(y)
    dT = isnothing(T) ? 0 : 1
    dZ = isnothing(Z) ? 0 : size(Z, 2)
    
    res = fill(NaN, n, 1 + dT + dZ)
    
    if vce == "nn"
        # Nearest neighbor variance estimation
        # This part in Python uses dups and dupsid which are from groupby transform
        # We need to find the matches neighbors for each point
        # For simplicity and performance, we can use a sliding window or NearestNeighbors.jl
        # But let's follow the logic: it finds the closest 'matches' observations
        
        for pos in 1:n
            rpos = dups[pos] - dupsid[pos]
            lpos = dupsid[pos] - 1
            
            while lpos + rpos < min(matches, n - 1)
                if pos - lpos - 1 < 1
                    rpos += dups[pos + rpos + 1]
                elseif pos + rpos + 1 > n
                    lpos += dups[pos - lpos - 1]
                elseif (X[pos] - X[pos - lpos - 1]) > (X[pos + rpos + 1] - X[pos])
                    rpos += dups[pos + rpos + 1]
                elseif (X[pos] - X[pos - lpos - 1]) < (X[pos + rpos + 1] - X[pos])
                    lpos += dups[pos - lpos - 1]
                else
                    rpos += dups[pos + rpos + 1]
                    lpos += dups[pos - lpos - 1]
                end
            end
            
            ind_J = max(1, pos - lpos):min(n, pos + rpos)
            y_J = sum(y[ind_J]) - y[pos]
            Ji = length(ind_J) - 1
            
            res[pos, 1] = sqrt(Ji / (Ji + 1)) * (y[pos] - y_J / Ji)
            
            if !isnothing(T)
                T_J = sum(T[ind_J]) - T[pos]
                res[pos, 2] = sqrt(Ji / (Ji + 1)) * (T[pos] - T_J / Ji)
            end
            
            if !isnothing(Z)
                for i in 1:dZ
                    Z_J = sum(Z[ind_J, i]) - Z[pos, i]
                    res[pos, 1 + dT + i] = sqrt(Ji / (Ji + 1)) * (Z[pos, i] - Z_J / Ji)
                end
            end
        end
    else
        w = if vce == "hc0"
            1.0
        elseif vce == "hc1"
            sqrt(n / (n - d))
        elseif vce == "hc2"
            1.0 ./ sqrt.(1.0 .- hii)
        else # hc3
            1.0 ./ (1.0 .- hii)
        end
        
        m_mat = reshape(m, n, :)
        res[:, 1] = w .* (y .- m_mat[:, 1])
        if dT == 1
            res[:, 2] = w .* (T .- m_mat[:, 2])
        end
        if dZ > 0
            for i in 1:dZ
                res[:, 1 + dT + i] = w .* (Z[:, i] .- m_mat[:, 1 + dT + i])
            end
        end
    end
    
    return res
end

function rdrobust_vce(d, s, RX, res, C)
    k = size(RX, 2)
    M = zeros(k, k)
    n = size(res, 1)
    
    if isnothing(C)
        w = 1.0
        if d == 0
            M = crossprod(reshape(res[:, 1], :, 1) .* RX)
        else
            for i in 1:(d+1)
                SS_i = reshape(res[:, i], :, 1) .* res
                for j in 1:(d+1)
                    M .+= crossprod(RX .* (s[i] * s[j] .* reshape(SS_i[:, j], :, 1)), RX)
                end
            end
        end
    else
        clusters = unique(C)
        g = length(clusters)
        w = ((n - 1) / (n - k)) * (g / (g - 1))
        
        if d == 0
            for i in 1:g
                ind = C .== clusters[i]
                Xi = RX[ind, :]
                ri = res[ind, 1]
                Xr = (Xi' * reshape(ri, :, 1))'
                M .+= Xr' * Xr
            end
        else
            for i in 1:g
                ind = C .== clusters[i]
                Xi = RX[ind, :]
                ri = res[ind, :]
                MHolder = zeros(1 + d, k)
                for l in 1:(d+1)
                    MHolder[l, :] = (Xi' * (s[l] .* reshape(ri[:, l], :, 1)))'
                end
                summedvalues = sum(MHolder, dims=1)
                M .+= summedvalues' * summedvalues
            end
        end
    end
    
    return w .* M
end

function rdrobust_bw(Y, X, T, Z, C, W, c, o, nu, o_B, h_V, h_B, scale, 
                    vce, nnmatch, kernel, dups, dupsid, covs_drop_coll)
    
    dT = isnothing(T) ? 0 : 1
    dZ = isnothing(Z) ? 0 : size(Z, 2)
    
    w = rdrobust_kweight(X, c, h_V, kernel)
    if !isnothing(W) && length(W) > 1
        w = W .* w
    end
    
    ind_V = w .> 0
    eY = Y[ind_V]
    eX = X[ind_V]
    eW = w[ind_V]
    n_V = sum(ind_V)
    
    D_V = copy(eY)
    R_V = fill(NaN, n_V, o + 1)
    for j in 0:o
        R_V[:, j + 1] = (eX .- c).^j
    end
    
    invG_V = qrXXinv(R_V .* sqrt.(eW))
    
    e_v = zeros(o + 1)
    e_v[nu + 1] = 1.0
    
    s = [1.0]
    eT = nothing
    eC = nothing
    eZ = nothing
    
    if !isnothing(T)
        eT = T[ind_V]
        D_V = hcat(D_V, eT)
    end
    
    if !isnothing(Z)
        eZ = Z[ind_V, :]
        D_V = hcat(D_V, eZ)
        
        U = (R_V .* eW)' * D_V
        ZWD = (eZ .* eW)' * D_V
        
        colsZ = (1 + dT + 1):size(D_V, 2)
        # In Julia, ranges are 1-based
        
        UiGU = U[:, colsZ]' * invG_V * U
        ZWZ = ZWD[:, colsZ] .- UiGU[:, colsZ]
        ZWY = ZWD[:, 1:(1+dT)] .- UiGU[:, 1:(1+dT)]
        
        gamma = if covs_drop_coll == 1
            pinv(ZWZ) * ZWY
        else
            ZWZ \ ZWY
        end
        s = vcat(1.0, -gamma[:, 1])
    end
    
    if !isnothing(C)
        eC = C[ind_V]
    end
    
    beta_V = invG_V * ((R_V .* eW)' * D_V)
    
    if isnothing(Z) && !isnothing(T)
        tau_Y = factorial(nu) * beta_V[nu + 1, 1]
        tau_T = factorial(nu) * beta_V[nu + 1, 2]
        s = [1/tau_T, -(tau_Y / tau_T^2)]
    end
    
    if !isnothing(Z) && !isnothing(T)
        s_T = vcat(1.0, -gamma[:, 2])
        colsZ_adj = [1; collect(colsZ)]
        beta_Y = beta_V[nu + 1, [1; collect(colsZ)]]
        tau_Y = factorial(nu) * (s' * beta_Y)
        beta_T = beta_V[nu + 1, [2; collect(colsZ)]]
        tau_T = factorial(nu) * (s_T' * beta_T)
        
        s = vcat(1/tau_T, -(tau_Y / tau_T^2), 
                 -(1/tau_T) .* gamma[:, 1] .+ (tau_Y / tau_T^2) .* gamma[:, 2])
    end
    
    dups_V = nothing
    dupsid_V = nothing
    hii = 0.0
    predicts_V = 0.0
    
    if vce == "nn"
        dups_V = dups[ind_V]
        dupsid_V = dupsid[ind_V]
    end
    
    if vce in ["hc0", "hc1", "hc2", "hc3"]
        predicts_V = R_V * beta_V
        if vce in ["hc2", "hc3"]
            hii = sum((R_V * invG_V) .* (R_V .* eW), dims=2)
        end
    end
    
    res_V = rdrobust_res(eX, eY, eT, eZ, predicts_V, hii, vce, nnmatch, dups_V, dupsid_V, o + 1)
    
    aux = rdrobust_vce(dT + dZ, s, R_V .* eW, res_V, eC)
    V_V = (invG_V * aux * invG_V)[nu + 1, nu + 1]
    
    v = (R_V .* eW)' * ((eX .- c) ./ h_V).^(o + 1)
    Hp = [h_V^j for j in 0:o]
    BConst = (Hp .* (invG_V * v))[nu + 1]
    
    w_B = rdrobust_kweight(X, c, h_B, kernel)
    if !isnothing(W) && length(W) > 1
        w_B = W .* w_B
    end
    
    ind_B = w_B .> 0
    n_B = sum(ind_B)
    eY_B = Y[ind_B]
    eX_B = X[ind_B]
    eW_B = w_B[ind_B]
    
    R_B = fill(NaN, n_B, o_B + 1)
    for j in 0:o_B
        R_B[:, j + 1] = (eX_B .- c).^j
    end
    
    invG_B = qrXXinv(R_B .* sqrt.(eW_B))
    
    eT_B = isnothing(T) ? nothing : T[ind_B]
    eZ_B = isnothing(Z) ? nothing : Z[ind_B, :]
    eC_B = isnothing(C) ? nothing : C[ind_B]
    
    D_B = eY_B
    if !isnothing(eT_B)
        D_B = hcat(D_B, eT_B)
    end
    if !isnothing(eZ_B)
        D_B = hcat(D_B, eZ_B)
    end
    
    beta_B = invG_B * ((R_B .* eW_B)' * D_B)
    
    BWreg = 0.0
    if scale > 0
        dups_B = nothing
        dupsid_B = nothing
        hii_B = 0.0
        predicts_B = 0.0
        
        if vce == "nn"
            dups_B = dups[ind_B]
            dupsid_B = dupsid[ind_B]
        end
        
        if vce in ["hc0", "hc1", "hc2", "hc3"]
            predicts_B = R_B * beta_B
            if vce in ["hc2", "hc3"]
                hii_B = sum((R_B * invG_B) .* (R_B .* eW_B), dims=2)
            end
        end
        
        res_B = rdrobust_res(eX_B, eY_B, eT_B, eZ_B, predicts_B, hii_B, vce, nnmatch, dups_B, dupsid_B, o_B + 1)
        aux_B = rdrobust_vce(dT + dZ, s, R_B .* eW_B, res_B, eC_B)
        V_B = (invG_B * aux_B * invG_B)[end, end]
        BWreg = 3 * BConst^2 * V_B
    end
    
    beta_aux = beta_B[end, :]
    B = sqrt(2 * (o + 1 - nu)) * BConst * (s' * beta_aux)
    V = (2 * nu + 1) * h_V^(2 * nu + 1) * V_V
    R = scale * (2 * (o + 1 - nu)) * BWreg
    rate = 1 / (2 * o + 3)
    
    return V, B, R, rate
end

end # module
