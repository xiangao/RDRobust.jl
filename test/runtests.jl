using RDRobust
using Test
using DataFrames
using CSV
using Statistics

@testset "RDRobust.jl" begin
    # Load senate data
    data_path = joinpath(@__DIR__, "rdrobust_senate.csv")
    if isfile(data_path)
        df = CSV.read(data_path, DataFrame)
        # In the senate data, 'margin' is the running variable (x)
        # 'vote' is the outcome (y)
        
        y = df.vote
        x = df.margin
        
        # Test basic rdrobust
        results = rdrobust(y, x)
        
        @test results isa RDRobustOutput
        @test results.N == [595, 702]
        @test isapprox(results.Estimate.tau_us[1], 7.414, atol=1e-3)
        
        # Test basic rdbwselect
        bw = rdbwselect(y, x)
        @test bw isa RDBWSelectOutput
        @test isapprox(bw.bws[1, :h_left], 17.754, atol=1e-3)
        
        # Test basic rdplot
        plot_data = rdplot(y, x)
        @test plot_data isa RDPlotOutput
        @test plot_data.J == [15, 35]
    else
        @warn "Senate data not found at $data_path, skipping some tests."
    end

    @testset "matches R rdrobust 4.0.0, with and without covariates" begin
        # reference values and the mortgages subset come from test/make_r_reference.R
        ref = CSV.read(joinpath(@__DIR__, "r_reference.csv"), DataFrame)
        sen = CSV.read(joinpath(@__DIR__, "rdrobust_senate.csv"), DataFrame)
        vet = CSV.read(joinpath(@__DIR__, "mortgages_subset.csv"), DataFrame)
        cs  = Float64.(coalesce.(Matrix(sen[:, [:class, :termshouse, :termssenate]]), NaN))  # rows with missing are dropped, as in R
        fits = Dict(
            "sharp"      => rdrobust(sen.vote, sen.margin),
            "sharp_covs" => rdrobust(sen.vote, sen.margin, covs = cs),
            "sharp_hc1"  => rdrobust(sen.vote, sen.margin, covs = cs, vce = "hc1"),
            "fuzzy"      => rdrobust(vet.home_ownership, vet.qob_minus_kw, fuzzy = vet.vet_wwko),
            "fuzzy_covs" => rdrobust(vet.home_ownership, vet.qob_minus_kw, fuzzy = vet.vet_wwko,
                                     covs = reshape(Float64.(vet.nonwhite), :, 1)),
        )
        for r in eachrow(ref)
            f = fits[r.case]
            E = f.Estimate
            @testset "$(r.case)" begin
                @test isapprox([E.tau_us[1], E.tau_bc[1], E.se_us[1], E.se_rb[1]],
                               [r.tau_us, r.tau_bc, r.se_us, r.se_rb]; rtol = 1e-6)
                @test isapprox([f.bws.h_left[1], f.bws.b_left[1]], [r.h, r.b]; rtol = 1e-6)
                @test f.N_h == [r.N_h_l, r.N_h_r]
            end
        end
    end

    @testset "warns when covariates duplicate the local polynomial" begin
        # x takes three values per side; z is a function of x that is not
        # linear on either side, so with p = 1 and an h that keeps all six
        # mass points the covariate-adjusted jump is not identified
        n = 1200
        x = repeat([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], n ÷ 6)
        z = hcat(Float64.(x .== -0.5), Float64.(x .== 1.5), Float64.(x .== -2.5) .+ Float64.(x .== 0.5))
        y = 0.3 .* (x .> 0) .+ 0.1 .* x .+ sin.(1:n)
        @test_logs (:warn, r"not\s+identified") match_mode = :any rdrobust(y, x; covs = z, h = 3.0, b = 3.0)
        @test_logs min_level = Base.CoreLogging.Warn rdrobust(y, x; covs = reshape(cos.(1:n), :, 1), h = 3.0, b = 3.0)
    end
end
