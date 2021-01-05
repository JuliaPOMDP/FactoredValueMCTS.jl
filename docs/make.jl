using FactoredValueMCTS
using Documenter

makedocs(;
    modules=[FactoredValueMCTS],
    authors="Stanford Intelligent Systems Laboratory",
    repo="https://github.com/JuliaPOMDP/FactoredValueMCTS.jl/blob/{commit}{path}#L{line}",
    sitename="FactoredValueMCTS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaPOMDP.github.io/FactoredValueMCTS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaPOMDP/FactoredValueMCTS.jl",
)
