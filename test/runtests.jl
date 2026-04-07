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
end
