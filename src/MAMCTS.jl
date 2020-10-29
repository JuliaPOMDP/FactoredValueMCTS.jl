module MAMCTS

using Random
using LinearAlgebra

using Parameters
using POMDPs
using MAPOMDPs
using POMDPPolicies
using POMDPLinter
using MCTS
using LightGraphs
using BeliefUpdaters

using MCTS: convert_estimator
import POMDPModelTools

### 
# Factored Value MCTS
#

abstract type CoordinationStatistics end

"""
Random Policy factored for each agent. Avoids exploding action space. 
"""
struct FactoredRandomPolicy{RNG<:AbstractRNG,P<:JointMDP, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U
end

FactoredRandomPolicy(problem::JointMDP; rng=Random.GLOBAL_RNG, updater=NothingUpdater()) = FactoredRandomPolicy(rng, problem, updater)

function POMDPs.action(policy::FactoredRandomPolicy, s)
    return [rand(policy.rng, get_agent_actions(policy.problem, i, si)) for (i, si) in enumerate(s)]
end

POMDPs.solve(solver::RandomSolver, problem::JointMDP) = FactoredRandomPolicy(solver.rng, problem, NothingUpdater())

include(joinpath("fvmcts", "fv_mcts_vanilla.jl"))
include(joinpath("fvmcts", "action_coordination", "varel.jl"))
include(joinpath("fvmcts", "action_coordination", "maxplus.jl"))

export
    FVMCTSSolver,
    MaxPlus,
    VarEl

###

###
# Naive Fully Connected Centralized MCTS
#

include(joinpath("fcmcts", "fcmcts.jl"))
export 
    FCMCTSSolver

###

end
