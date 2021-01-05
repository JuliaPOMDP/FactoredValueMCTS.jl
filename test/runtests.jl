using FactoredValueMCTS
using Test

using POMDPs
using MultiAgentSysAdmin
using MultiUAVDelivery

@testset "FactoredValueMCTS.jl" begin

    @testset "varel" begin
        @testset "sysadmin" begin

            @testset "local" begin
                mdp = BiSysAdmin()
                solver = FVMCTSSolver()
                planner = solve(solver, mdp)
                s = rand(initialstate(mdp))
                a = action(planner, s)
                @test a isa actiontype(mdp)
            end
            

            @testset "global" begin
                mdp = BiSysAdmin(;global_rewards=true)
                solver = FVMCTSSolver()
                planner = solve(solver, mdp)
                s = rand(initialstate(mdp))
                a = action(planner, s)
                @test a isa actiontype(mdp)
            end
        end
    end    

    @testset "maxplus" begin
        @testset "sysadmin" begin

            @testset "local" begin
                mdp = BiSysAdmin()
                solver = FVMCTSSolver(;coordination_strategy=MaxPlus())
                planner = solve(solver, mdp)
                s = rand(initialstate(mdp))
                a = action(planner, s)
                @test a isa actiontype(mdp)
            end
        

            @testset "global" begin
                mdp = BiSysAdmin(;global_rewards=true)
                solver = FVMCTSSolver(;coordination_strategy=MaxPlus())
                planner = solve(solver, mdp)
                s = rand(initialstate(mdp))
                a = action(planner, s)
                @test a isa actiontype(mdp)
            end
        end

        @testset "uav" begin
            mdp = FirstOrderMultiUAVDelivery()
            solver = FVMCTSSolver(;coordination_strategy=MaxPlus())
            planner = solve(solver, mdp)
            s = rand(initialstate(mdp))
            a = action(planner, s)
            @test a isa actiontype(mdp)
        end

    end

end
