# FactoredValueMCTS

[![CI](https://github.com/JuliaPOMDP/FactoredValueMCTS.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaPOMDP/FactoredValueMCTS.jl/actions/workflows/ci.yml)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/FactoredValueMCTS.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/FactoredValueMCTS.jl?branch=master)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliapomdp.github.io/FactoredValueMCTS.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliapomdp.github.io/FactoredValueMCTS.jl/dev)

This package implements the Monte Carlo Tree Search (MCTS) planning algorithm for Multi-Agent MDPs. The algorithm factorizes the true action value function, based on the locality of interactions between agents that is encoded with a Coordination Graph. We implement two schemes for coordinating the actions for the team of agents during the MCTS computations. The first is the iterative message-passing MaxPlus, while the second is the exact Variable Elimination. We thus get two different Factored Value MCTS algorithms, FV-MCTS-MaxPlus and FV-MCTS-VarEl respectively.

The full FV-MCTS-MaxPlus algorithm is described in our AAMAS 2021 paper _Scalable Anytime Planning for Multi-Agent MDPs_ ([Arxiv](https://arxiv.org/abs/2101.04788)). The FV-MCTS-Varel is based on the Factored Statistics algorithm from the AAAI 2015 paper _Scalable Planning and Learning from Multi-Agent POMDPs_ ([Extended Version](https://arxiv.org/abs/1404.1140)) applied to Multi-Agent MDPs rather than POMDPs. We use the latter as a baseline and show how the former outperforms it on two distinct simulated domains.

To use our solver, the domain must implement the interface from [MultiAgentPOMDPs.jl](https://github.com/JuliaPOMDP/MultiAgentPOMDPs.jl). For examples, please see [MultiAgentSysAdmin](https://github.com/JuliaPOMDP/MultiAgentSysAdmin.jl) and [MultiUAVDelivery](https://github.com/JuliaPOMDP/MultiUAVDelivery.jl), which are the two domains from our AAMAS 2021 paper. Experiments from the paper are available at https://github.com/rejuvyesh/FVMCTS_experiments.

## Installation

```julia
using Pkg
Pkg.add("FactoredValueMCTS")
```
## Citation

```
@inproceedings{choudhury2021scalable,
    title={Scalable Anytime Planning for Multi-Agent {MDP}s},
    author={Shushman Choudhury and Jayesh K Gupta and Peter Morales and Mykel J Kochenderfer},
    booktitle={International Conference on Autonomous Agents and MultiAgent Systems},
    year={2021}
}
```