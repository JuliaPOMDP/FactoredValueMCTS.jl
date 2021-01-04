module MAMCTS

using Random
using LinearAlgebra

using POMDPs
using MultiAgentPOMDPs
using POMDPPolicies
using POMDPLinter: @req, @subreq, @POMDP_require
using MCTS
using LightGraphs
using BeliefUpdaters

using MCTS: convert_estimator
import POMDPModelTools

using POMDPSimulators: RolloutSimulator
import POMDPs

function POMDPs.simulate(sim::RolloutSimulator, mdp::JointMDP, policy::Policy, initialstate::S) where {S}

    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end

    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initialstate

    disc = 1.0
    r_total = zeros(n_agents(mdp))
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s)

        sp, r = @gen(:sp, :r)(mdp, s, a, sim.rng)

        r_total .+= disc.*r

        s = sp

        disc *= discount(mdp)
        step += 1
    end

    return r_total
end


### 
# Factored Value MCTS
#

abstract type CoordinationStatistics end

include(joinpath("fvmcts", "factoredpolicy.jl"))
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
