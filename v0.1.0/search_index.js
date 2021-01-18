var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FactoredValueMCTS","category":"page"},{"location":"#FactoredValueMCTS","page":"Home","title":"FactoredValueMCTS","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FactoredValueMCTS]","category":"page"},{"location":"#FactoredValueMCTS.FVMCTSSolver","page":"Home","title":"FactoredValueMCTS.FVMCTSSolver","text":"Factored Value Monte Carlo Tree Search solver datastructure\n\nFields:     n_iterations::Int64         Number of iterations during each action() call.         default: 100\n\nmax_time::Float64\n    Maximum CPU time to spend computing an action.\n    default::Inf\n\ndepth::Int64\n    Number of iterations during each action() call.\n    default: 100\n\nexploration_constant::Float64:\n    Specifies how much the solver should explore. In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.\n    The exploration terms for FV-MCTS-Var-El and FV-MCTS-Max-Plus are different but the role of c is the same.\n    default: 1.0\n\nrng::AbstractRNG:\n    Random number generator\n\nestimate_value::Any (rollout policy)\n    Function, object, or number used to estimate the value at the leaf nodes.\n    If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.\n    If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.\n    If this is a number, the value will be set to that number\n    default: RolloutEstimator(RandomSolver(rng))\n\ninit_Q::Any\n    Function, object, or number used to set the initial Q(s,a) value at a new node.\n    If this is a function `f`, `f(mdp, s, a)` will be called to set the value.\n    If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.\n    If this is a number, Q will be set to that number\n    default: 0.0\n\ninit_N::Any\n    Function, object, or number used to set the initial N(s,a) value at a new node.\n    If this is a function `f`, `f(mdp, s, a)` will be called to set the value.\n    If this is an object `o`, `init_N(o, mdp, s, a)` will be called.\n    If this is a number, N will be set to that number\n    default: 0\n\nreuse_tree::Bool\n    If this is true, the tree information is re-used for calculating the next plan.\n    Of course, clear_tree! can always be called to override this.\n    default: false\n\ncoordination_strategy::AbstractCoordinationStrategy\n    The specific strategy with which to compute the best joint action from the current MCTS statistics.\n    default: VarEl()\n\n\n\n\n\n","category":"type"},{"location":"#FactoredValueMCTS.FactoredRandomPolicy","page":"Home","title":"FactoredValueMCTS.FactoredRandomPolicy","text":"Random Policy factored for each agent. Avoids exploding action space. \n\n\n\n\n\n","category":"type"},{"location":"#FactoredValueMCTS.MaxPlusStatistics","page":"Home","title":"FactoredValueMCTS.MaxPlusStatistics","text":"Tracks the specific informations and statistics we need to use Max-Plus to coordinateaction the joint action in Factored-Value MCTS. Putting parameters here is a little ugly but coordinateaction can't have them since VarEl doesn't use those args.\n\nFields:     adjmatgraph::SimpleGraph         The coordination graph as a LightGraphs SimpleGraph.\n\nmessage_iters::Int64\n    Number of rounds of message passing.\n\nmessage_norm::Bool\n    Whether to normalize the messages or not after message passing.\n\nuse_agent_utils::Bool\n    Whether to include the per-agent utilities while computing the best agent action (see our paper for details)\n\nnode_exploration::Bool\n    Whether to use the per-node UCB style bonus while computing the best agent action (see our paper for details)\n\nedge_exploration::Bool\n    Whether to use the per-edge UCB style bonus after the message passing rounds (see our paper for details). One of this or node_exploration MUST be true for exploration.\n\nall_states_stats::Dict{AbstractVector{S},PerStateMPStats}\n    Maps each joint state in the tree to the per-state statistics.\n\n\n\n\n\n","category":"type"},{"location":"#FactoredValueMCTS.VarElStatistics","page":"Home","title":"FactoredValueMCTS.VarElStatistics","text":"Tracks the specific informations and statistics we need to use Var-El to coordinate_action the joint action in Factored-Value MCTS.\n\nFields:     coordgraphcomponents::Vector{Vector{Int64}}         The list of coordination graph components, i.e., cliques, where each element is a list of agent IDs that are in a mutual clique.\n\nmin_degree_ordering::Vector{Int64}\n    Ordering of agent IDs in increasing CG degree. This ordering is the heuristic most typically used for the elimination order in Var-El.\n\nn_component_stats::Dict{AbstractVector{S},Vector{Vector{Int64}}}\n    Maps each joint state in the tree (for which we need to compute the UCB action) to the frequency of each component's various local actions.\n\nq_component_stats::Dict{AbstractVector{S},Vector{Vector{Float64}}}\n    Maps each joint state in the tree to the accumulated q-value of each component's various local actions.\n\n\n\n\n\n","category":"type"},{"location":"#FactoredValueMCTS.coordinate_action-Union{Tuple{A}, Tuple{S}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.MaxPlusStatistics{S}},S}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.MaxPlusStatistics{S}},S,Float64}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.MaxPlusStatistics{S}},S,Float64,Int64}} where A where S","page":"Home","title":"FactoredValueMCTS.coordinate_action","text":"Runs Max-Plus at the current state using the per-state MaxPlusStatistics to compute the best joint action with either or both of node-wise and edge-wise exploration bonus. Rounds of message passing are followed by per-node maximization.\n\n\n\n\n\n","category":"method"},{"location":"#FactoredValueMCTS.coordinate_action-Union{Tuple{A}, Tuple{S}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.VarElStatistics{S}},S}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.VarElStatistics{S}},S,Float64}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.VarElStatistics{S}},S,Float64,Int64}} where A where S","page":"Home","title":"FactoredValueMCTS.coordinate_action","text":"Runs variable elimination at the current state using the VarEl Statistics to compute the best joint action with the component-wise exploration bonus. FYI: Rather complicated.\n\n\n\n\n\n","category":"method"},{"location":"#FactoredValueMCTS.maxplus_joint_mcts_planner-Union{Tuple{A}, Tuple{S}, Tuple{FVMCTSSolver,MultiAgentPOMDPs.JointMDP{S,A},S,Int64,Bool,Bool,Bool,Bool}} where A where S","page":"Home","title":"FactoredValueMCTS.maxplus_joint_mcts_planner","text":"Called internally in solve() to create the FVMCTSPlanner where Max-Plus is the specific action coordination strategy. Creates MaxPlusStatistics and assumes the various MP flags are sent down from the CoordinationStrategy object given to the solver.\n\n\n\n\n\n","category":"method"},{"location":"#FactoredValueMCTS.update_statistics!-Union{Tuple{A}, Tuple{S}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.MaxPlusStatistics{S}},S,A,AbstractArray{Float64,1}}} where A where S","page":"Home","title":"FactoredValueMCTS.update_statistics!","text":"Take the q-value from the MCTS step and distribute the updates across the per-node and per-edge q-stats as per the formula in our paper.\n\n\n\n\n\n","category":"method"},{"location":"#FactoredValueMCTS.update_statistics!-Union{Tuple{A}, Tuple{S}, Tuple{MultiAgentPOMDPs.JointMDP{S,A},FactoredValueMCTS.FVMCTSTree{S,A,FactoredValueMCTS.VarElStatistics{S}},S,A,AbstractArray{Float64,1}}} where A where S","page":"Home","title":"FactoredValueMCTS.update_statistics!","text":"Take the q-value from the MCTS step and distribute the updates across the component q-stats as per the formula in the Amato-Oliehoek paper.\n\n\n\n\n\n","category":"method"},{"location":"#FactoredValueMCTS.varel_joint_mcts_planner-Union{Tuple{A}, Tuple{S}, Tuple{FVMCTSSolver,MultiAgentPOMDPs.JointMDP{S,A},S}} where A where S","page":"Home","title":"FactoredValueMCTS.varel_joint_mcts_planner","text":"Called internally in solve() to create the FVMCTSPlanner where Var-El is the specific action coordination strategy. Creates VarElStatistics internally with the CG components and the minimum degree ordering heuristic.\n\n\n\n\n\n","category":"method"}]
}
