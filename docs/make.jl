using Documenter, RDRobust

makedocs(
    sitename = "RDRobust.jl",
    modules  = [RDRobust],
    pages = [
        "Home"      => "index.md",
        "Reference" => "reference.md",
    ],
    warnonly = true,
    remotes  = nothing,
)

deploydocs(
    repo       = "github.com/xiangao/RDRobust.jl.git",
    devbranch  = "master",
    push_preview = false,
)
