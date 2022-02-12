using Distributions
using ImportanceWeightedRiskMetrics
using ParallelTreeSampling
using POMDPModelTools
using POMDPPolicies
using POMDPs
using POMDPSimulators
using Random

include("tree_mdp.jl")


# Creates an instance of TreeMDP.
function create_tree_amdp(amdp, distribution; reduction="sum")
    return TreeMDP(amdp, 1.0, [], [], distribution, reduction)
end


# Runs Monte Carlo baseline.
function run_baseline(mdp, fixed_s, disturbance; N=1000)
    costs = [sum(collect(simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> rand(disturbance(mdp, s))), fixed_s)[:r]))
             for _ in 1:N]
    output = (costs, [])
    return output
end


# Runs MCTS-based sampling.
function run_mcts(tree_mdp, fixed_s; N=1000, c=0.3, vloss=0.0, α=0.1, β=1.0, γ=0.0)
    solver = PISSolver(; depth=100,
                       estimate_value=rollout,  # Required.
                       exploration_constant=c,
                       n_iterations=N,
                       enable_state_pw=false,   # Required.
                       show_progress=true,
                       tree_in_info=true,
                       virtual_loss=vloss,
                       α=α);
    planner = solve(solver, tree_mdp);
    a, info = action_info(planner, TreeState(fixed_s); tree_in_info=true, β=β, γ=γ)
    output = (planner.mdp.costs, [], planner.mdp.IS_weights, info)
    return output, planner
end


# Returns risk metrics.
function eval_metrics(costs; weights=nothing, alpha=1e-4)
    costs = [Float64(cost) for cost in costs]
    if weights == nothing
        weights = ones(length(costs))
    else
        weights = [Float64(weight) for weight in weights]
    end

    metrics = IWRiskMetrics(costs, weights, alpha);
    return metrics
end
