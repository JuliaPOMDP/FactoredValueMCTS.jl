using FactoredValueMCTS
using Documenter

makedocs(;
    sitename="FactoredValueMCTS.jl",    
    authors="Stanford Intelligent Systems Laboratory",
    modules=[FactoredValueMCTS],
    format=Documenter.HTML()
)

deploydocs(;
    repo="github.com/JuliaPOMDP/FactoredValueMCTS.jl"
)
