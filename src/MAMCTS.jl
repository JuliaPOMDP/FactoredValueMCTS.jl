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
