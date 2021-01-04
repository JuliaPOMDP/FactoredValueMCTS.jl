

Base.@kwdef mutable struct FCMCTSSolver <: AbstractMCTSSolver
    n_iterations::Int64 = 100
    max_time::Float64 = Inf
    depth::Int64 = 10
    exploration_constant::Float64 = 1.0
    rng::AbstractRNG = Random.GLOBAL_RNG
    estimate_value::Any = RolloutEstimator(RandomSolver(rng))
    init_Q::Any = 0.0
    init_N::Any = 0
    reuse_tree::Bool = false
end

mutable struct FCMCTSTree{S,A}
    # To track if state node in tree already
    # NOTE: We don't strictly need this at all if no tree reuse...
    state_map::Dict{S,Int64}

    # these vectors have one entry for each state node
    # Only doing factored satistics (for actions), not state components
    child_ids::Vector{Vector{Int}}
    total_n::Vector{Int}
    s_labels::Vector{S}

    # TODO(jkg): is this the best way to track stats?
    # these vectors have one entry for each action node
    n::Vector{Int64}
    q::Vector{Float64}
    a_labels::Vector{A}

    lock::ReentrantLock
end

function FCMCTSTree{S,A}(init_state::S, lock::ReentrantLock, sz::Int=1000) where {S,A}
    sz = min(sz, 100_000)
    return FCMCTSTree{S,A}(Dict{S,Int64}(),
                           sizehint!(Vector{Int}[], sz),
                           sizehint!(Int[], sz),
                           sizehint!(typeof(init_state)[], sz),
                           Int64[],
                           Float64[],
                           sizehint!(Vector{A}[], sz),
                           lock)
end

Base.isempty(t::FCMCTSTree) = isempty(t.state_map)
state_nodes(t::FCMCTSTree) = (FCStateNode(t, id) for id in 1:length(t.total_n))

struct FCStateNode{S,A}
    tree::FCMCTSTree{S,A}
    id::Int64
end

# accessors for state nodes
@inline state(n::FCStateNode) = lock(n.tree.lock) do
    n.tree.s_labels[n.id]
end
@inline total_n(n::FCStateNode) = n.tree.total_n[n.id]
@inline children(n::FCStateNode) = (FCActionNode(n.tree, id) for id in n.tree.child_ids[n.id])

# Adding action node info
struct FCActionNode{S,A}
    tree::FCMCTSTree{S,A}
    id::Int64
end

# accessors for action nodes
@inline POMDPs.action(n::FCActionNode) = n.tree.a_labels[n.id]


mutable struct FCMCTSPlanner{S,A,SE,RNG<:AbstractRNG} <: AbstractMCTSPlanner{JointMDP{S,A}}
    solver::FCMCTSSolver
    mdp::JointMDP{S,A}
    tree::FCMCTSTree{S,A}
    solved_estimate::SE
    rng::RNG
end

function FCMCTSPlanner(solver::FCMCTSSolver, mdp::JointMDP{S,A}) where {S,A}
    init_state = initialstate(mdp, solver.rng)
    tree = FCMCTSTree{S,A}(init_state, ReentrantLock(), solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return FCMCTSPlanner(solver, mdp, tree, se, solver.rng)
end


function clear_tree!(planner::FCMCTSPlanner)
    lock(planner.tree.lock) do
        # Clear out state hash dict entirely
        empty!(planner.tree.state_map)
        # Empty state vectors with state hints
        sz = min(planner.solver.n_iterations, 100_000)

        empty!(planner.tree.s_labels)
        sizehint!(planner.tree.s_labels, sz)

        empty!(planner.tree.child_ids)
        sizehint!(planner.tree.child_ids, sz)
        empty!(planner.tree.total_n)
        sizehint!(planner.tree.total_n, sz)

        empty!(planner.tree.n)
        empty!(planner.tree.q)
        empty!(planner.tree.a_labels)
    end
end

function POMDPs.solve(solver::FCMCTSSolver, mdp::JointMDP)
    return FCMCTSPlanner(solver, mdp)
end

function POMDPs.action(planner::FCMCTSPlanner, s)
    clear_tree!(planner)
    plan!(planner, s)
    s_lut = lock(planner.tree.lock) do
        planner.tree.state_map[s]
    end
    best_anode = lock(planner.tree.lock) do
        compute_best_action_node(planner.mdp, planner.tree, FCStateNode(planner.tree, s_lut)) # c = 0.0 by default
    end

    best_a = lock(planner.tree.lock) do
        action(best_anode)
    end
    return best_a
end

function POMDPModelTools.action_info(planner::FCMCTSPlanner, s)
    a = POMDPs.action(planner, s)
    return a, nothing
end


function plan!(planner::FCMCTSPlanner, s)
    planner.tree = build_tree(planner, s)
end

function build_tree(planner::FCMCTSPlanner, s::AbstractVector{S}) where S
    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    root = insert_node!(planner.tree, planner, s)
    # build the tree
    @sync for n = 1:n_iterations
        @spawn simulate(planner, root, depth)
    end
    return planner.tree
end

function simulate(planner::FCMCTSPlanner, node::FCStateNode, depth::Int64)

    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree

    # once depth is zero return
    if isterminal(planner.mdp, s)
        return 0.0
    elseif depth == 0
        return sum(estimate_value(planner.solved_estimate, planner.mdp, s, depth))
    end

    # Choose best UCB action (NOT an action node)
    ucb_action_node = lock(planner.tree.lock) do
        compute_best_action_node(mdp, planner.tree, node, planner.solver.exploration_constant)
    end
    ucb_action = lock(planner.tree.lock) do
        action(ucb_action_node)
    end

    # @show ucb_action
    # MC Transition
    sp, r = @gen(:sp, :r)(mdp, s, ucb_action, rng)

    # NOTE(jkg): just summing up the rewards to get a single value
    # TODO: should we divide by n_agents?
    r = sum(r)

    spid = lock(tree.lock) do
        get(tree.state_map, sp, 0) # may be non-zero even with no tree reuse
    end
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        # TODO estimate_value
        # NOTE(jkg): again just summing up the values to get a single value
        q = r + discount(mdp) * sum(estimate_value(planner.solved_estimate, planner.mdp, sp, depth - 1))
    else
        q = r + discount(mdp) * simulate(planner, FCStateNode(tree, spid) , depth - 1)
    end

    ## Not bothering with tree vis right now
    # Augment N(s)
    lock(tree.lock) do
        tree.total_n[node.id] += 1
        tree.n[ucb_action_node.id] += 1
        tree.q[ucb_action_node.id] += (q - tree.q[ucb_action_node.id]) / tree.n[ucb_action_node.id]
    end

    return q
end

# NOTE: This is a bit different from https://github.com/JuliaPOMDP/MCTS.jl/blob/master/src/vanilla.jl#L328
function insert_node!(tree::FCMCTSTree, planner::FCMCTSPlanner, s)

    lock(tree.lock) do
        push!(tree.s_labels, s)
        tree.state_map[s] = length(tree.s_labels)
        push!(tree.child_ids, [])
    end

    # NOTE: Doing state-dep actions here the JointMDP way
    state_dep_jtactions = vec(map(collect, Iterators.product((agent_actions(planner.mdp, i, si) for (i, si) in enumerate(s))...)))
    total_n = 0

    for a in state_dep_jtactions
        n = init_N(planner.solver.init_N, planner.mdp, s, a)
        total_n += n
        lock(tree.lock) do
            push!(tree.n, n)
            push!(tree.q, init_Q(planner.solver.init_Q, planner.mdp, s, a))
            push!(tree.a_labels, a)
            push!(last(tree.child_ids), length(tree.n))
        end
    end
    lock(tree.lock) do
        push!(tree.total_n, total_n)
    end
    ln = lock(tree.lock) do
        length(tree.total_n)
    end
    return FCStateNode(tree, ln)
end



# NOTE: The logic here is a bit simpler than https://github.com/JuliaPOMDP/MCTS.jl/blob/master/src/vanilla.jl#L390
# Double check that this is still the behavior we want
function compute_best_action_node(mdp::JointMDP, tree::FCMCTSTree, node::FCStateNode, c::Float64=0.0)
    best_val = -Inf # The Q value
    best = first(children(node))

    sn = total_n(node)

    child_nodes = children(node)

    for sanode in child_nodes

        val = tree.q[sanode.id] + c*sqrt(log(sn + 1)/ (tree.n[sanode.id] + 1))


        if val > best_val
            best_val = val
            best = sanode
        end
    end
    return best
end

@POMDP_require simulate(planner::FCMCTSPlanner, s, depth::Int64) begin
    mdp = planner.mdp
    P = typeof(mdp)
    @assert P <: JointMDP       # req does different thing?
    SV = statetype(P)
    @req iterate(::SV)
    #@assert typeof(SV) <: AbstractVector # TODO: Is this correct?
    AV = actiontype(P)
    @assert typeof(AV) <: AbstractVector
    @req discount(::P)
    @req isterminal(::P, ::SV)
    @subreq insert_node!(planner.tree, planner, s)
    @subreq estimate_value(planner.solved_estimate, mdp, s, depth)
    @req gen(::P, ::SV, ::AV, ::typeof(planner.rng)) # XXX this is not exactly right - it could be satisfied with transition

    # MMDP reqs
    @req agent_actions(::P, ::Int64)
    @req agent_actions(::P, ::Int64, ::eltype(SV)) # TODO should this be eltype?
    @req n_agents(::P)

    # TODO: Should we also have this requirement for SV?
    @req isequal(::S, ::S)
    @req hash(::S)
end

@POMDP_require insert_node!(tree::FCMCTSTree, planner::FCMCTSPlanner, s) begin

    P = typeof(planner.mdp)
    AV = actiontype(P)
    A = eltype(AV)
    SV = typeof(s)
    S = eltype(SV)

    # TODO: Review IQ and IN
    # Should this be ::S or ::SV? We can have global state that's not a vector.
    IQ = typeof(planner.solver.init_Q)
    if !(IQ <: Number) && !(IQ <: Function)
        @req init_Q(::IQ, ::P, ::S, ::Vector{Int64}, ::AbstractVector{A})
    end

    IN = typeof(planner.solver.init_N)
    if !(IN <: Number) && !(IN <: Function)
        @req init_N(::IQ, ::P, ::S, ::Vector{Int64}, ::AbstractVector{A})
    end

    @req isequal(::S, ::S)
    @req hash(::S)
end
