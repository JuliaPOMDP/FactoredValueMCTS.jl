mutable struct VarElStatistics{S} <: CoordinationStatistics
    coord_graph_components::Vector{Vector{Int64}}
    min_degree_ordering::Vector{Int64}
    n_component_stats::Dict{AbstractVector{S},Vector{Vector{Int64}}}
    q_component_stats::Dict{AbstractVector{S},Vector{Vector{Float64}}}
end

function clear_statistics!(ve_stats::VarElStatistics)
    empty!(ve_stats.n_component_stats)
    empty!(ve_stats.q_component_stats)
end


function coordinate_action(mdp::JointMDP{S,A}, tree::JointMCTSTree{S,A,VarElStatistics{S}}, s::AbstractVector{S},
                           exploration_constant::Float64=0.0, node_id::Int64=0) where {S,A}

    n_agents = length(s)
    best_action_idxs = MVector{n_agents}([-1 for i in 1:n_agents])

    #
    # !Note: Acquire lock so as to avoid race
    #
    state_q_stats = lock(tree.lock) do
        tree.coordination_stats.q_component_stats[s]
    end
    state_n_stats = lock(tree.lock) do
        tree.coordination_stats.n_component_stats[s]
    end
    state_total_n = lock(tree.lock) do
        (node_id > 0) ? tree.total_n[node_id] : 0
    end

    # Maintain set of potential functions
    # NOTE: Hashing a vector here
    potential_fns = Dict{Vector{Int64},Vector{Float64}}()
    for (comp, q_stats) in zip(tree.coordination_stats.coord_graph_components, state_q_stats)
        potential_fns[comp] = q_stats
    end

    # Need this for reverse process
    # Maps agent to other elements in best response functions and corresponding set of actions
    # E.g. Agent 2 -> (3,4) in its best response and corresponding vector of agent 2 best actions
    best_response_fns = Dict{Int64,Tuple{Vector{Int64},Vector{Int64}}}()

    state_dep_actions = [get_agent_actions(mdp, i, si) for (i, si) in enumerate(s)]

    # Iterate over variable ordering
    # Need to maintain intermediate tables
    for ag_idx in tree.coordination_stats.min_degree_ordering

        # Lookup factors with agent in them and simultaneously construct
        # members of new potential function, and delete old factors
        agent_factors = Vector{Vector{Int64}}(undef, 0)
        new_potential_members = Vector{Int64}(undef, 0)
        for k in collect(keys(potential_fns))
            if ag_idx in k

                # Agent to-be-eliminated is in factor
                push!(agent_factors, k)

                # Construct key for new potential as union of all others
                # except ag_idx
                for ag in k
                    if ag != ag_idx && ~(ag in new_potential_members)
                        push!(new_potential_members, ag)
                    end
                end
            end
        end

        if isempty(new_potential_members) == true
            # No out neighbors..either at beginning or end of ordering
            @assert agent_factors == [[ag_idx]] "agent_factors $(agent_factors) is not just [ag_idx] $([ag_idx])!"
            best_action_idxs[ag_idx] = _best_actionindex_empty(potential_fns,
                                                               state_dep_actions,
                                                               tree.all_agent_actions,
                                                               ag_idx)

        else

            # Generate new potential function
            # AND the best response vector for eliminated agent
            n_comp_actions = prod([length(tree.all_agent_actions[c]) for c in new_potential_members])

            # NOTE: Tuples should ALWAYS use tree.all_agent_actions for indexing
            comp_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in new_potential_members)

            # Initialize q-stats for new potential and best response action vector
            # will be inserted into corresponding dictionaries at the end
            new_potential_stats = Vector{Float64}(undef, n_comp_actions)
            best_response_vect = Vector{Int64}(undef, n_comp_actions)

            # Iterate over new potential joint actions and compute new payoff and best response
            for comp_ac_idx = 1:n_comp_actions

                # Get joint action for other members in potential
                ct_idx = CartesianIndices(comp_tup)[comp_ac_idx]

                # For maximizing over agent actions
                # As before, we now need to init with -Inf
                ag_ac_values = zeros(length(tree.all_agent_actions[ag_idx]))

                # TODO: Agent actions should already be in order
                # Only do anything if action legal
                for (ag_ac_idx, ag_ac) in enumerate(tree.all_agent_actions[ag_idx])

                    if ag_ac in state_dep_actions[ag_idx]

                        # Need to look up corresponding stats from agent_factors
                        for factor in agent_factors

                            # NOTE: Need to reconcile the ORDER of ag_idx in factor
                            factor_action_idxs = MVector{length(factor),Int64}(undef)

                            for (idx, f) in enumerate(factor)

                                # if f is ag_idx, set corresponding factor action to ag_ac
                                if f == ag_idx
                                    factor_action_idxs[idx] = ag_ac_idx
                                else
                                    # Lookup index for corresp. agent action in ct_idx
                                    new_pot_idx = findfirst(isequal(f), new_potential_members)
                                    factor_action_idxs[idx] = ct_idx[new_pot_idx]
                                end # f == ag_idx
                            end

                            # NOW we can look up the stats of the factor
                            factor_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in factor)
                            factor_action_linidx = LinearIndices(factor_tup)[factor_action_idxs...]

                            ag_ac_values[ag_ac_idx] += potential_fns[factor][factor_action_linidx]

                            # Additionally add exploration stats if factor in original set
                            factor_comp_idx = findfirst(isequal(factor), tree.coordination_stats.coord_graph_components)
                            if state_total_n > 0 && ~(isnothing(factor_comp_idx)) # NOTE: Julia1.1
                                ag_ac_values[ag_ac_idx] += exploration_constant * sqrt((log(state_total_n+1.0))/(state_n_stats[factor_comp_idx][factor_action_linidx]+1.0))
                            end
                        end # factor in agent_factors
                    else
                        ag_ac_values[ag_ac_idx] = -Inf
                    end # ag_ac in state_dep_actions
                end # ag_ac_idx = 1:length(tree.all_agent_actions[ag_idx])


                # Now we lookup ag_ac_values for the best value to be put in new_potential_stats
                # and the best index to be put in best_response_vect
                # NOTE: The -Inf mask should ensure only legal idxs chosen
                # If all ag_ac_values equal, should we sample randomly?
                best_val, best_idx = findmax(ag_ac_values)

                new_potential_stats[comp_ac_idx] = best_val
                best_response_vect[comp_ac_idx] = best_idx
            end # comp_ac_idx in n_comp_actions

            # Finally, we enter new stats vector and best response vector back to dicts
            potential_fns[new_potential_members] = new_potential_stats
            best_response_fns[ag_idx] = (new_potential_members, best_response_vect)
        end # isempty(new_potential_members)

        # Delete keys in agent_factors from potential fns since variable has been eliminated
        for factor in agent_factors
            delete!(potential_fns, factor)
        end
    end # ag_idx in min_deg_ordering

    # NOTE: At this point, best_action_idxs has at least one entry...for the last action obtained
    @assert !all(isequal(-1), best_action_idxs) "best_action_idxs is still undefined!"

    # Message passing in reverse order to recover best action
    for ag_idx in Base.Iterators.reverse(tree.coordination_stats.min_degree_ordering)

        # Only do something if best action already not obtained
        if best_action_idxs[ag_idx] == -1

            # Should just be able to lookup best response function
            (agents, best_response_vect) = best_response_fns[ag_idx]

            # Members of agents should already have their best action defined
            agent_ac_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in agents)
            best_agents_action_idxs = [best_action_idxs[ag] for ag in agents]
            best_response_idx = LinearIndices(agent_ac_tup)[best_agents_action_idxs...]

            # Assign best action for ag_idx
            best_action_idxs[ag_idx] = best_response_vect[best_response_idx]
        end # isdefined
    end

    # Finally, return best action by iterating over best action indices
    # NOTE: best_action should use state-dep actions to reverse index
    best_action = [tree.all_agent_actions[ag][idx] for (ag, idx) in enumerate(best_action_idxs)]

    return best_action
end


function update_statistics!(mdp::JointMDP{S,A}, tree::JointMCTSTree{S,A,VarElStatistics{S}},
                            s::AbstractVector{S}, ucb_action::AbstractVector{A}, q::AbstractVector{Float64}) where {S,A}

    n_agents = length(s)

    for (idx, comp) in enumerate(tree.coordination_stats.coord_graph_components)

        # Create cartesian index tuple
        comp_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in comp)

        # RECOVER local action corresp. to ucb action
        # TODO: Review this carefully. Need @req for action index for agent.
        local_action = [ucb_action[c] for c in comp]
        local_action_idxs = [get_agent_actionindex(mdp, c, a) for (a, c) in zip(local_action, comp)]

        comp_ac_idx = LinearIndices(comp_tup)[local_action_idxs...]

        # NOTE: NOW we can update stats. Could generalize incremental update more here
        lock(tree.lock) do
            tree.coordination_stats.n_component_stats[s][idx][comp_ac_idx] += 1
            q_comp_value = sum(q[c] for c in comp)
            tree.coordination_stats.q_component_stats[s][idx][comp_ac_idx] +=
                (q_comp_value - tree.coordination_stats.q_component_stats[s][idx][comp_ac_idx]) / tree.coordination_stats.n_component_stats[s][idx][comp_ac_idx]
        end
    end
end


function init_statistics!(tree::JointMCTSTree{S,A,VarElStatistics{S}}, planner::JointMCTSPlanner,
                          s::AbstractVector{S}) where {S,A}

    n_comps = length(tree.coordination_stats.coord_graph_components)
    n_component_stats = Vector{Vector{Int64}}(undef, n_comps)
    q_component_stats = Vector{Vector{Float64}}(undef, n_comps)

    n_agents = length(s)

    # TODO: Could actually make actions state-dependent if need be
    for (idx, comp) in enumerate(tree.coordination_stats.coord_graph_components)

        n_comp_actions = prod([length(tree.all_agent_actions[c]) for c in comp])

        n_component_stats[idx] = Vector{Int64}(undef, n_comp_actions)
        q_component_stats[idx] = Vector{Float64}(undef, n_comp_actions)

        comp_tup = Tuple(1:length(tree.all_agent_actions[c]) for c in comp)

        for comp_ac_idx = 1:n_comp_actions

            # Generate action subcomponent and call init_Q and init_N for it
            ct_idx = CartesianIndices(comp_tup)[comp_ac_idx] # Tuple corresp to
            local_action = [tree.all_agent_actions[c][ai] for (c, ai) in zip(comp, Tuple(ct_idx))]

            # NOTE: init_N and init_Q are functions of component AND local action
            # TODO(jkg): init_N and init_Q need to be defined
            n_component_stats[idx][comp_ac_idx] = init_N(planner.solver.init_N, planner.mdp, s, comp, local_action)
            q_component_stats[idx][comp_ac_idx] = init_Q(planner.solver.init_Q, planner.mdp, s, comp, local_action)
        end
    end

    # Update tree member
    lock(tree.lock) do
        tree.coordination_stats.n_component_stats[s] = n_component_stats
        tree.coordination_stats.q_component_stats[s] = q_component_stats
    end
end

@inline function _best_actionindex_empty(potential_fns, state_dep_actions, all_agent_actions, ag_idx)
    # NOTE: This is inefficient but necessary for state-dep actions?
    if length(state_dep_actions[ag_idx]) == length(all_agent_actions[ag_idx])
        _, best_ac_idx = findmax(potential_fns[[ag_idx]])
    else
        # Now we need to choose the best index from among legal actions
        # Create an array with illegal actions having -Inf and then fill legal vals
        # TODO: More efficient way to do this?
        masked_action_vals = fill(-Inf, length(all_agent_actions[ag_idx]))
        for (iac, ac) in enumerate(all_agent_actions[ag_idx])
            if ac in state_dep_actions[ag_idx]
                masked_action_vals[iac] = potential_fns[[ag_idx]][iac]
            end
        end
        _, best_ac_idx = findmax(masked_action_vals)
    end
    return best_ac_idx
end
