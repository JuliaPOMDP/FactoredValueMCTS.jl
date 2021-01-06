# FactoredValueMCTS

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaPOMDP.github.io/FactoredValueMCTS.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaPOMDP.github.io/FactoredValueMCTS.jl/dev)
[![Build Status](https://github.com/JuliaPOMDP/FactoredValueMCTS.jl/workflows/CI/badge.svg)](https://github.com/JuliaPOMDP/FactoredValueMCTS.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaPOMDP/FactoredValueMCTS.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/FactoredValueMCTS.jl)

This package implements the Monte Carlo Tree Search (MCTS) planning algorithm for Multi-Agent MDPs. The algorithm factorizes the true action value function, based on the locality of interactions between agents that is encoded with a Coordination Graph. We implement two schemes for coordinating the actions for the team of agents during the MCTS computations. The first is the iterative message-passing MaxPlus, while the second is the exact Variable Elimination. We thus get two different Factored Value MCTS algorithms, FV-MCTS-MaxPlus and FV-MCTS-VarEl respectively.

The full FV-MCTS-MaxPlus algorithm is described in our AAMAS 2021 paper _Scalable Anytime Planning for Multi-Agent MDPs_ (link pending). The FV-MCTS-Varel is based on the Factored Statistics algorithm from the AAAI 2015 paper _Scalable Planning and Learning from Multi-Agent POMDPs_ ([Extended Version](https://arxiv.org/abs/1404.1140)) applied to Multi-Agent MDPs rather than POMDPs. We use the latter as a baseline and show how the former outperforms it on two distinct simulated domains.

To use our solver, the domain must implement the interface from [MultiAgentPOMDPs.jl](https://github.com/JuliaPOMDP/MultiAgentPOMDPs.jl). For examples, please see [MultiAgentSysAdmin](https://github.com/JuliaPOMDP/MultiAgentSysAdmin.jl) and [MultiUAVDelivery](https://github.com/JuliaPOMDP/MultiUAVDelivery.jl), which are the two domains from our AAMAS 2021 paper.
