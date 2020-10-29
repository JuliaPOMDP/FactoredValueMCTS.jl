using MAMCTS
using Documenter

makedocs(;
    modules=[MAMCTS],
    authors="Stanford Intelligent Systems Laboratory",
    repo="https://github.com/rejuvyesh/MAMCTS.jl/blob/{commit}{path}#L{line}",
    sitename="MAMCTS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rejuvyesh.github.io/MAMCTS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rejuvyesh/MAMCTS.jl",
)
