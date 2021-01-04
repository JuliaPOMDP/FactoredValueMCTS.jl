# NOTE: Matrix implicitly assumes all agents have same number of actions
mutable struct PerStateMPStats
    agent_action_n::Matrix{Int64} # N X A
    agent_action_q::Matrix{Float64}
    edge_action_n::Matrix{Int64} # |E| X A^2
    edge_action_q::Matrix{Float64}
end

"""
Tracks the specific informations and statistics we need to use Max-Plus to coordinate_action
the joint action in Factored-Value MCTS. Putting parameters here is a little ugly but coordinate_action can't have them since VarEl doesn't use those args.

Fields:
    adjmatgraph::SimpleGraph
        The coordination graph as a LightGraphs SimpleGraph.
    
    message_iters::Int64
        Number of rounds of message passing.

    message_norm::Bool
        Whether to normalize the messages or not after message passing.

    use_agent_utils::Bool
        Whether to include the per-agent utilities while computing the best agent action (see our paper for details)

    node_exploration::Bool
        Whether to use the per-node UCB style bonus while computing the best agent action (see our paper for details)

    edge_exploration::Bool
        Whether to use the per-edge UCB style bonus after the message passing rounds (see our paper for details). One of this or node_exploration MUST be true for exploration.

    all_states_stats::Dict{AbstractVector{S},PerStateMPStats}
        Maps each joint state in the tree to the per-state statistics.
"""
mutable struct MaxPlusStatistics{S} <: CoordinationStatistics
    adjmatgraph::SimpleGraph
    message_iters::Int64
    message_norm::Bool
    use_agent_utils::Bool
    node_exploration::Bool
    edge_exploration::Bool # NOTE: One of this or node exploration must be true
    all_states_stats::Dict{S,PerStateMPStats}
end

function clear_statistics!(mp_stats::MaxPlusStatistics)
    empty!(mp_stats.all_states_stats)
end

function update_statistics!(mdp::JointMDP{S,A}, tree::FVMCTSTree{S,A,MaxPlusStatistics{S}},
                            s::S, ucb_action::A, q::AbstractFloat) where {S,A}

    update_statistics!(mdp, tree, s, ucb_action, ones(typeof(q), n_agents(mdp)) * q)
end

"""
Take the q-value from the MCTS step and distribute the updates across the per-node and per-edge q-stats as per the formula in our paper.
"""
function update_statistics!(mdp::JointMDP{S,A}, tree::FVMCTSTree{S,A,MaxPlusStatistics{S}},
                            s::S, ucb_action::A, q::AbstractVector{Float64}) where {S,A}

    state_stats = tree.coordination_stats.all_states_stats[s]
    nagents = n_agents(mdp)

    # Update per agent action stats
    for i = 1:nagents
        ac_idx = agent_actionindex(mdp, i, ucb_action[i])
        lock(tree.lock) do
            state_stats.agent_action_n[i, ac_idx] += 1
            state_stats.agent_action_q[i, ac_idx] +=
                (q[i] - state_stats.agent_action_q[i, ac_idx]) / state_stats.agent_action_n[i, ac_idx]
        end
    end

    # Now update per-edge action stats
    for (idx, e) in enumerate(edges(tree.coordination_stats.adjmatgraph))
        # NOTE: Need to be careful about action ordering
        # Being more general to have unequal agent actions
        edge_comp = (e.src,e.dst)
        edge_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in edge_comp)
        edge_ac_idx = LinearIndices(edge_tup)[agent_actionindex(mdp, e.src, ucb_action[e.src]),
                                              agent_actionindex(mdp, e.dst, ucb_action[e.dst])]
        q_edge_value = q[e.src] + q[e.dst]

        lock(tree.lock) do
            state_stats.edge_action_n[idx, edge_ac_idx] += 1
            state_stats.edge_action_q[idx, edge_ac_idx] +=
                (q_edge_value - state_stats.edge_action_q[idx, edge_ac_idx]) / state_stats.edge_action_n[idx, edge_ac_idx]
        end
    end

    lock(tree.lock) do
        tree.coordination_stats.all_states_stats[s] = state_stats
    end

end

function init_statistics!(tree::FVMCTSTree{S,A,MaxPlusStatistics{S}}, planner::FVMCTSPlanner,
                          s::S) where {S,A}

    n_agents = length(s)

    # NOTE: Assuming all agents have the same actions here
    n_all_actions = length(tree.all_agent_actions[1])

    agent_action_n = zeros(Int64, n_agents, n_all_actions)
    agent_action_q = zeros(Float64, n_agents, n_all_actions)

    # Loop over agents and then actions
    # TODO: Need to define init_N and init_Q for single agent
    for i = 1:n_agents
        for (j, ac) in enumerate(tree.all_agent_actions[i])
            agent_action_n[i, j] = init_N(planner.solver.init_N, planner.mdp, s, i, ac)
            agent_action_q[i, j] = init_Q(planner.solver.init_Q, planner.mdp, s, i, ac)
        end
    end

    n_edges = ne(tree.coordination_stats.adjmatgraph)
    edge_action_n = zeros(Int64, n_edges, n_all_actions^2)
    edge_action_q = zeros(Float64, n_edges, n_all_actions^2)

    # Loop over edges and then action_i \times action_j
    for (idx, e) in enumerate(edges(tree.coordination_stats.adjmatgraph))
        edge_comp = (e.src, e.dst)
        n_edge_actions = prod([length(tree.all_agent_actions[c]) for c in edge_comp])
        edge_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in edge_comp)

        for edge_ac_idx = 1:n_edge_actions
            ct_idx = CartesianIndices(edge_tup)[edge_ac_idx]
            edge_action = [tree.all_agent_actions[c] for c in edge_comp]

            edge_action_n[idx, edge_ac_idx] = init_N(planner.solver.init_N, planner.mdp, s, edge_comp, edge_action)
            edge_action_q[idx, edge_ac_idx] = init_Q(planner.solver.init_Q, planner.mdp, s, edge_comp, edge_action)
        end
    end

    state_stats = PerStateMPStats(agent_action_n, agent_action_q, edge_action_n, edge_action_q)
    lock(tree.lock) do
        tree.coordination_stats.all_states_stats[s] = state_stats
    end
end

"""
Runs Max-Plus at the current state using the per-state MaxPlusStatistics to compute the best joint action with either or both of node-wise and edge-wise exploration bonus. Rounds of message passing are followed by per-node maximization.
"""
function coordinate_action(mdp::JointMDP{S,A}, tree::FVMCTSTree{S,A,MaxPlusStatistics{S}}, s::S,
                           exploration_constant::Float64=0.0, node_id::Int64=0) where {S,A}

    state_stats = lock(tree.lock) do
        tree.coordination_stats.all_states_stats[s]
    end
    adjgraphmat = lock(tree.lock) do
        tree.coordination_stats.adjmatgraph
    end
    k = tree.coordination_stats.message_iters
    message_norm = tree.coordination_stats.message_norm

    n_agents = length(s)
    state_agent_actions = [agent_actions(mdp, i, si) for (i, si) in enumerate(s)]
    n_all_actions = length(tree.all_agent_actions[1])
    n_edges = ne(tree.coordination_stats.adjmatgraph)

    # Init forward and backward messages and q0
    fwd_messages = zeros(Float64, n_edges, n_all_actions)
    bwd_messages = zeros(Float64, n_edges, n_all_actions)

    if tree.coordination_stats.use_agent_utils
        q_values = state_stats.agent_action_q / n_agents
    else
        q_values = zeros(size(state_stats.agent_action_q))
    end

    state_total_n = lock(tree.lock) do
        (node_id > 0) ? tree.total_n[node_id] : 1
    end


    # Iterate over passes
    for t = 1:k
        fnormdiff, bnormdiff = perform_message_passing!(fwd_messages, bwd_messages, mdp, tree.all_agent_actions,
                                    adjgraphmat, state_agent_actions, n_edges, q_values, message_norm,
                                    0, state_stats, state_total_n)

        if !tree.coordination_stats.use_agent_utils
            q_values = zeros(size(state_stats.agent_action_q))
        end

        # Update Q value with messages
        for i = 1:n_agents

            # need indices of all edges that agent is involved in
            nbrs = neighbors(tree.coordination_stats.adjmatgraph, i)

            edgelist = collect(edges(adjgraphmat))

            if tree.coordination_stats.use_agent_utils
                @views q_values[i, :] = state_stats.agent_action_q[i, :]/n_agents
            end
            for n in nbrs
                if Edge(i,n) in edgelist # use backward message
                    q_values[i,:] += bwd_messages[findfirst(isequal(Edge(i,n)), edgelist), :]
                elseif Edge(n,i) in edgelist
                    q_values[i,:] += fwd_messages[findfirst(isequal(Edge(n,i)), edgelist), :]
                else
                    @warn "Neither edge found!"
                end
            end
        end

        # If converged, break
        if isapprox(fnormdiff, 0.0) && isapprox(bnormdiff, 0.0)
            break
        end
    end # for t = 1:k

    # If edge exploration flag enabled, do a final exploration bonus
    if tree.coordination_stats.edge_exploration
        perform_message_passing!(fwd_messages, bwd_messages, mdp, tree.all_agent_actions,
                                 adjgraphmat, state_agent_actions, n_edges, q_values, message_norm,
                                 exploration_constant, state_stats, state_total_n)
    end # if edge_exploration


    # Maximize q values for agents
    best_action = Vector{A}(undef, n_agents)
    for i = 1:n_agents

        # NOTE: Again can't just iterate over agent actions as it may be a subset
        exp_q_values = zeros(length(state_agent_actions[i]))
        if tree.coordination_stats.node_exploration
            for (idx, ai) in enumerate(state_agent_actions[i])
                ai_idx = agent_actionindex(mdp, i, ai)
                exp_q_values[idx] = q_values[i, ai_idx] + exploration_constant*sqrt((log(state_total_n + 1.0))/(state_stats.agent_action_n[i, ai_idx] + 1.0))
            end
        else
            for (idx, ai) in enumerate(state_agent_actions[i])
                ai_idx = agent_actionindex(mdp, i, ai)
                exp_q_values[idx] = q_values[i, ai_idx]
            end
        end

        # NOTE: Can now look up index in exp_q_values and then again look at state_agent_actions
        _, idx = findmax(exp_q_values)
        best_action[i] = state_agent_actions[i][idx]
    end

    return best_action
end

function perform_message_passing!(fwd_messages::AbstractArray{F,2}, bwd_messages::AbstractArray{F,2},
                                  mdp, all_agent_actions,
                                  adjgraphmat, state_agent_actions, n_edges::Int, q_values, message_norm,
                                  exploration_constant, state_stats, state_total_n) where {F}
    # Iterate over edges
    fwd_messages_old = deepcopy(fwd_messages)
    bwd_messages_old = deepcopy(bwd_messages)
    for (e_idx, e) in enumerate(edges(adjgraphmat))

        i = e.src
        j = e.dst
        edge_tup_indices = LinearIndices(Tuple(1:length(all_agent_actions[c]) for c in (i,j)))

        # forward: maximize sender
        # NOTE: Can't do enumerate as action set might be smaller
        # Need to look up global index of agent action and use that
        # Need to break up vectorized loop
        @inbounds for aj in state_agent_actions[j]
            aj_idx = agent_actionindex(mdp, j, aj)
            fwd_message_vals = zeros(length(state_agent_actions[i]))
            # TODO: Should we use inbounds here again?
            @inbounds for (idx, ai) in enumerate(state_agent_actions[i])
                ai_idx = agent_actionindex(mdp, i, ai)
                fwd_message_vals[idx] = q_values[i, ai_idx] - bwd_messages_old[e_idx, ai_idx] + state_stats.edge_action_q[e_idx, edge_tup_indices[ai_idx, aj_idx]]/n_edges + exploration_constant * sqrt( (log(state_total_n + 1.0)) / (state_stats.edge_action_n[e_idx, edge_tup_indices[ai_idx, aj_idx]] + 1) )
            end
            fwd_messages[e_idx, aj_idx] = maximum(fwd_message_vals)
        end

        @inbounds for ai in state_agent_actions[i]
            ai_idx = agent_actionindex(mdp, i, ai)
            bwd_message_vals = zeros(length(state_agent_actions[j]))
            @inbounds for (idx, aj) in enumerate(state_agent_actions[j])
                aj_idx = agent_actionindex(mdp, j, aj)
                bwd_message_vals[idx] = q_values[j, aj_idx] - fwd_messages_old[e_idx, aj_idx] + state_stats.edge_action_q[e_idx, edge_tup_indices[ai_idx, aj_idx]]/n_edges + exploration_constant * sqrt( (log(state_total_n + 1.0))/ (state_stats.edge_action_n[e_idx, edge_tup_indices[ai_idx, aj_idx]] + 1) )
            end
            bwd_messages[e_idx, ai_idx] = maximum(bwd_message_vals)
        end

        # Normalize messages for better convergence
        if message_norm
            @views fwd_messages[e_idx, :] .-= sum(fwd_messages[e_idx, :])/length(fwd_messages[e_idx, :])
            @views bwd_messages[e_idx, :] .-= sum(bwd_messages[e_idx, :])/length(bwd_messages[e_idx, :])
        end

    end # (idx,edges) in enumerate(edges)

    # Return norm of message difference
    return norm(fwd_messages - fwd_messages_old), norm(bwd_messages - bwd_messages_old)
end
