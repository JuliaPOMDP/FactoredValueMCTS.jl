module FactoredValueMCTS

using Random
using LinearAlgebra

using POMDPs
using POMDPTools
using MultiAgentPOMDPs
using POMDPLinter: @req, @subreq, @POMDP_require
using MCTS
using Graphs
using MCTS: convert_estimator


# Patch simulate to support vector of rewards
function POMDPs.simulate(sim::RolloutSimulator, mdp::JointMDP, policy::Policy, initialstate::S) where {S}

    if sim.eps === nothing
        eps = 0.0
    else
        eps = sim.eps
    end

    if sim.max_steps === nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initialstate

    
    # TODO: doesn't this add unnecessary action search?
    r = @gen(:r)(mdp, s, action(policy, s), sim.rng)
    if r isa AbstractVector
        r_total = zeros(n_agents(mdp))
    else
        r_total = 0.0
    end
    sim_helper!(r_total, sim, mdp, policy, s, max_steps, eps)
    return r_total
end

function sim_helper!(r_total::AbstractVector{F}, sim, mdp, policy, s, max_steps, eps) where {F}
    step = 1
    disc = 1.0
    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s)

        sp, r = @gen(:sp, :r)(mdp, s, a, sim.rng)

        r_total .+= disc.*r

        s = sp

        disc *= discount(mdp)
        step += 1
    end

end

function sim_helper!(r_total::AbstractFloat, sim, mdp, policy, s, max_steps, eps)
    step = 1
    disc = 1.0
    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s)

        sp, r = @gen(:sp, :r)(mdp, s, a, sim.rng)

        r_total += disc.*r

        s = sp

        disc *= discount(mdp)
        step += 1
    end

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


end
