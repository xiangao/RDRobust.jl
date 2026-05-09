module RDRobust

include("utils.jl")
include("rdbwselect.jl")
include("rdrobust_impl.jl")
include("rdplot.jl")

using .Utils
using .BandwidthSelection
using .RDRobustEstimation
using .RDPlotEstimation

export rdbwselect, rdrobust, rdplot
export RDBWSelectOutput, RDRobustOutput, RDPlotOutput

end # module
