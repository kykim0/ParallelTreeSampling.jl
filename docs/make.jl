using ParallelTreeSampling
using Documenter

DocMeta.setdocmeta!(ParallelTreeSampling, :DocTestSetup, :(using ParallelTreeSampling); recursive=true)

makedocs(;
    modules=[ParallelTreeSampling],
    authors="Kyu-Young Kim <kykim@cs.stanford.edu> and contributors",
    repo="https://github.com/kykim0/ParallelTreeSampling.jl/blob/{commit}{path}#{line}",
    sitename="ParallelTreeSampling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kykim0.github.io/ParallelTreeSampling.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kykim0/ParallelTreeSampling.jl",
    devbranch="main",
)
