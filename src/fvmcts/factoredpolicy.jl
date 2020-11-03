
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