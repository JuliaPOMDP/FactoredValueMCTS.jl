using StaticArrays
using Parameters
using Base.Threads: @spawn

abstract type AbstractCoordinationStrategy end

struct VarEl <: AbstractCoordinationStrategy
end

Base.@kwdef struct MaxPlus <:AbstractCoordinationStrategy
    message_iters::Int64 = 10
    message_norm::Bool = true
    use_agent_utils::Bool = false
    node_exploration::Bool = true
    edge_exploration::Bool = true
end

"""
Factored Value Monte Carlo Tree Search solver datastructure

Fields:
    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum CPU time to spend computing an action.
        default::Inf
    
    depth::Int64
        Number of iterations during each action() call.
        default: 100
    
    exploration_constant::Float64:
        Specifies how much the solver should explore. In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        The exploration terms for FV-MCTS-Var-El and FV-MCTS-Max-Plus are different but the role of c is the same.
        default: 1.0
    
    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will be set to that number
        default: 0.0
    
    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will be set to that number
        default: 0
    
    reuse_tree::Bool
        If this is true, the tree information is re-used for calculating the next plan.
        Of course, clear_tree! can always be called to override this.
        default: false
    
    coordination_strategy::AbstractCoordinationStrategy
        The specific strategy with which to compute the best joint action from the current MCTS statistics.
        default: VarEl() 
"""
Base.@kwdef mutable struct FVMCTSSolver <: AbstractMCTSSolver
    n_iterations::Int64 = 100
    max_time::Float64 = Inf
    depth::Int64 = 10
    exploration_constant::Float64 = 1.0
    rng::AbstractRNG = Random.GLOBAL_RNG
    estimate_value::Any = RolloutEstimator(RandomSolver(rng))
    init_Q::Any = 0.0
    init_N::Any = 0
    reuse_tree::Bool = false
    coordination_strategy::AbstractCoordinationStrategy = VarEl()
end


mutable struct FVMCTSTree{S,A,CS<:CoordinationStatistics}

    # To map the multi-agent state vector to the ID of the node in the tree
    state_map::Dict{S,Int64}

    # The next two vectors have one for each node ID in the tree
    total_n::Vector{Int}                # The number of times the node has been tried
    s_labels::Vector{S} # The state corresponding to the node ID

    # List of all individual actions of each agent for coordination purposes.
    all_agent_actions::Vector{A}

    coordination_stats::CS
    lock::ReentrantLock
end

function FVMCTSTree(all_agent_actions::Vector{A},
                    coordination_stats::CS,
                    init_state::S,
                    lock::ReentrantLock,
                    sz::Int64=10000) where {S, A, CS <: CoordinationStatistics}

    return FVMCTSTree{S,A,CS}(Dict{S,Int64}(),
                                 sizehint!(Int[], sz),
                                 sizehint!(S[], sz),
                                 all_agent_actions,
                                 coordination_stats,
                                 lock
                                 )
end # function



Base.isempty(t::FVMCTSTree) = isempty(t.state_map)
state_nodes(t::FVMCTSTree) = (FVStateNode(t, id) for id in 1:length(t.total_n))

struct FVStateNode{S}
    tree::FVMCTSTree{S}
    id::Int64
end


# Accessors for state nodes
@inline state(n::FVStateNode) = n.tree.s_labels[n.id]
@inline total_n(n::FVStateNode) = n.tree.total_n[n.id]

## No need for `children` or ActionNode just yet

mutable struct FVMCTSPlanner{S, A, SE, CS <: CoordinationStatistics, RNG <: AbstractRNG} <: AbstractMCTSPlanner{JointMDP{S,A}}
    solver::FVMCTSSolver
    mdp::JointMDP{S,A}
    tree::FVMCTSTree{S,A,CS}
    solved_estimate::SE
    rng::RNG
end

"""
Called internally in solve() to create the FVMCTSPlanner where Var-El is the specific action coordination strategy.
Creates VarElStatistics internally with the CG components and the minimum degree ordering heuristic.
"""
function varel_joint_mcts_planner(solver::FVMCTSSolver,
                                  mdp::JointMDP{S,A},
                                  init_state::S,
                                  ) where {S,A}

    # Get coordination graph components from maximal cliques
    #adjmat = coord_graph_adj_mat(mdp)
    #@assert size(adjmat)[1] == n_agents(mdp) "Adjacency Matrix does not match number of agents!"

    #adjmatgraph = SimpleGraph(adjmat)
    adjmatgraph = coordination_graph(mdp)
    
    coord_graph_components = maximal_cliques(adjmatgraph)
    min_degree_ordering = sortperm(degree(adjmatgraph))

    # Initialize full agent actions
    all_agent_actions = Vector{(actiontype(mdp))}(undef, n_agents(mdp))
    for i = 1:n_agents(mdp)
        all_agent_actions[i] = agent_actions(mdp, i)
    end

    ve_stats = VarElStatistics{S}(coord_graph_components, min_degree_ordering,
                                                   Dict{typeof(init_state),Vector{Vector{Int64}}}(),
                                                   Dict{typeof(init_state),Vector{Vector{Int64}}}(),
                                                   )

    # Create tree from the current state
    tree = FVMCTSTree(all_agent_actions, ve_stats,
                         init_state, ReentrantLock(), solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)

    return FVMCTSPlanner(solver, mdp, tree, se, solver.rng)
end # end function

"""
Called internally in solve() to create the FVMCTSPlanner where Max-Plus is the specific action coordination strategy.
Creates MaxPlusStatistics and assumes the various MP flags are sent down from the CoordinationStrategy object given to the solver.
"""
function maxplus_joint_mcts_planner(solver::FVMCTSSolver,
                                    mdp::JointMDP{S,A},
                                    init_state::S,
                                    message_iters::Int64,
                                    message_norm::Bool,
                                    use_agent_utils::Bool,
                                    node_exploration::Bool,
                                    edge_exploration::Bool,
                                    ) where {S,A}

    @assert (node_exploration || edge_exploration) "At least one of nodes or edges should explore!"

#=     adjmat = coord_graph_adj_mat(mdp)
    @assert size(adjmat)[1] == n_agents(mdp) "Adjacency Mat does not match number of agents!" =#

    #adjmatgraph = SimpleGraph(adjmat)
    adjmatgraph = coordination_graph(mdp)
    @assert size(adjacency_matrix(adjmatgraph))[1] == n_agents(mdp)
    
    # Initialize full agent actions
    # TODO(jkg): this is incorrect? Or we need to override actiontype to refer to agent actions?
    all_agent_actions = Vector{(actiontype(mdp))}(undef, n_agents(mdp))
    for i = 1:n_agents(mdp)
        all_agent_actions[i] = agent_actions(mdp, i)
    end

    mp_stats = MaxPlusStatistics{S}(adjmatgraph,
                                                    message_iters,
                                                    message_norm,
                                                    use_agent_utils,
                                                    node_exploration,
                                                    edge_exploration,
                                                    Dict{S,PerStateMPStats}())

    # Create tree from the current state
    tree = FVMCTSTree(all_agent_actions, mp_stats,
                      init_state, ReentrantLock(), solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)

    return FVMCTSPlanner(solver, mdp, tree, se, solver.rng)
end


# Reset tree.
function clear_tree!(planner::FVMCTSPlanner)

    # Clear out state map dict entirely
    empty!(planner.tree.state_map)

    # Empty state vectors with state hints
    sz = min(planner.solver.n_iterations, 100_000)

    empty!(planner.tree.s_labels)
    sizehint!(planner.tree.s_labels, planner.solver.n_iterations)

    # Don't touch all_agent_actions and coord graph component
    # Just clear comp stats dict
    clear_statistics!(planner.tree.coordination_stats)
end

MCTS.init_Q(n::Number, mdp::JointMDP, s, c, a) = convert(Float64, n)
MCTS.init_N(n::Number, mdp::JointMDP, s, c, a) = convert(Int, n)


# No computation is done in solve; the solver is just given the mdp model that it will work with
# and in case of MaxPlus, the various flags for the MaxPlus behavior
function POMDPs.solve(solver::FVMCTSSolver, mdp::JointMDP)
    if typeof(solver.coordination_strategy) == VarEl
        return varel_joint_mcts_planner(solver, mdp, initialstate(mdp, solver.rng))
    elseif typeof(solver.coordination_strategy) == MaxPlus
        return maxplus_joint_mcts_planner(solver, mdp, initialstate(mdp, solver.rng),
                                          solver.coordination_strategy.message_iters,
                                          solver.coordination_strategy.message_norm,
                                          solver.coordination_strategy.use_agent_utils,
                                          solver.coordination_strategy.node_exploration,
                                          solver.coordination_strategy.edge_exploration)
    else
        throw(error("Not Implemented"))
    end
end


# IMP: Overriding action for FVMCTSPlanner here
# NOTE: Hardcoding no tree reuse for now
function POMDPs.action(planner::FVMCTSPlanner, s)
    clear_tree!(planner) # Always call this at the top
    plan!(planner, s)
    action =  coordinate_action(planner.mdp, planner.tree, s)
    return action
end

function POMDPModelTools.action_info(planner::FVMCTSPlanner, s)
    clear_tree!(planner) # Always call this at the top
    plan!(planner, s)
    action = coordinate_action(planner.mdp, planner.tree, s)
    return action, nothing
end


function plan!(planner::FVMCTSPlanner, s)
    planner.tree = build_tree(planner, s)
end

# build_tree can be called on the assumption that no reuse AND tree is reinitialized
function build_tree(planner::FVMCTSPlanner, s::S) where S

    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    root = insert_node!(planner.tree, planner, s)
    
    # Simulate can be multi-threaded
    @sync for n = 1:n_iterations
        @spawn simulate(planner, root, depth)
    end
    return planner.tree
end

function simulate(planner::FVMCTSPlanner, node::FVStateNode, depth::Int64)

    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree


    # once depth is zero return
    if isterminal(planner.mdp, s)
        return 0.0
    elseif depth == 0
        return estimate_value(planner.solved_estimate, planner.mdp, s, depth)
    end

    # Choose best UCB action (NOT an action node as in vanilla MCTS)
    ucb_action = coordinate_action(mdp, planner.tree, s, planner.solver.exploration_constant, node.id)

    # Monte Carlo Transition
    sp, r = @gen(:sp, :r)(mdp, s, ucb_action, rng)

    spid = lock(tree.lock) do
        get(tree.state_map, sp, 0) # may be non-zero even with no tree reuse
    end
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id

        q = r .+ discount(mdp) * estimate_value(planner.solved_estimate, planner.mdp, sp, depth - 1)
    else
        q = r .+ discount(mdp) * simulate(planner, FVStateNode(tree, spid) , depth - 1)
    end

    # NOTE: Not bothering with tree visualization right now
    # Augment N(s)
    lock(tree.lock) do
        tree.total_n[node.id] += 1
    end

    # Update component statistics! (non-trivial)
    # This is related but distinct from initialization
    update_statistics!(mdp, tree, s, ucb_action, q)

    return q
end

@POMDP_require simulate(planner::FVMCTSPlanner, s, depth::Int64) begin
    mdp = planner.mdp
    P = typeof(mdp)
    @assert P <: JointMDP
    #SV = statetype(P)
    #@assert typeof(SV) <: AbstractVector
    AV = actiontype(P)
    @assert typeof(A) <: AbstractVector
    @req discount(::P)
    @req isterminal(::P, ::SV)
    @subreq insert_node!(planner.tree, planner, s)
    @subreq estimate_value(planner.solved_estimate, mdp, s, depth)
    @req gen(::DDNOut{(:sp, :r)}, ::P, ::SV, ::A, ::typeof(planner.rng))

    ## Requirements from MMDP Model
    @req agent_actions(::P, ::Int64)
    @req agent_actions(::P, ::Int64, ::eltype(SV))
    @req n_agents(::P)
    @req coordination_graph(::P)

    # TODO: Should we also have this requirement for SV?
    @req isequal(::S, ::S)
    @req hash(::S)
end



function insert_node!(tree::FVMCTSTree{S,A,CS}, planner::FVMCTSPlanner,
                      s::S) where {S,A,CS <: CoordinationStatistics}

    lock(tree.lock) do
        push!(tree.s_labels, s)
        tree.state_map[s] = length(tree.s_labels)
        push!(tree.total_n, 1)

        # NOTE: Could actually make actions state-dependent if need be
        init_statistics!(tree, planner, s)
    end
    
    # length(tree.s_labels) is just an alias for the number of state nodes
    ls = lock(tree.lock) do
        length(tree.s_labels)
    end
    return FVStateNode(tree, ls)
end

@POMDP_require insert_node!(tree::FVMCTSTree, planner::FVMCTSPlanner, s) begin

    P = typeof(planner.mdp)
    AV = actiontype(P)
    A = eltype(AV)
    SV = typeof(s)
    #S = eltype(SV)

    # TODO: Review IQ and IN
    IQ = typeof(planner.solver.init_Q)
    if !(IQ <: Number) && !(IQ <: Function)
        @req init_Q(::IQ, ::P, ::SV, ::Vector{Int64}, ::AbstractVector{A})
    end

    IN = typeof(planner.solver.init_N)
    if !(IN <: Number) && !(IN <: Function)
        @req init_N(::IQ, ::P, ::SV, ::Vector{Int64}, ::AbstractVector{A})
    end

    @req isequal(::S, ::S)
    @req hash(::S)
end
