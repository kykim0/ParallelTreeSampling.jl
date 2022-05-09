using Distributions
using Random

using ImportanceWeightedRiskMetrics
using POMDPGym
using POMDPModelTools
using POMDPPolicies
using POMDPs
using POMDPSimulators

using ParallelTreeSampling


# Runs Monte Carlo baseline.
function run_mc(amdp::MDP, init_s, distrib_fn::Function; N=1000)
    function_policy = FunctionPolicy((s) -> rand(distrib_fn(amdp, s)))
    costs = [sum(collect(simulate(HistoryRecorder(), amdp, function_policy, init_s)[:r]))
             for _ in 1:N]
    output = (costs, [])
    return output
end


function run_mc(rmdp::RMDP, init_s, px; N=1000)
    noise_fn = (s) -> rand(px)  # Random noise.
    function_policy = FunctionPolicy(noise_fn)
    costs = [sum(collect(simulate(HistoryRecorder(), rmdp, function_policy, init_s)[:r]))
             for _ in 1:N]
    output = (costs,)
    return output
end


# Runs MCTS-based sampling.
function run_mcts(mdp::MDP, s, nominal_distrib_fn, a_selection;
                  N=1000, c=0.3, vloss=0.0, α=0.1, β=1.0, γ=0.0,
                  show_progress=true)
    solver = PISSolver(; depth=100,
                       exploration_constant=c,
                       n_iterations=N,
                       enable_action_pw=false,  # Needed for discrete cases.
                       k_state=Inf,             # Needed for discrete cases (to always transition).
                       virtual_loss=vloss,
                       nominal_distrib_fn=nominal_distrib_fn,
                       action_selection=a_selection,
                       show_progress=show_progress)
    planner = solve(solver, mdp)
    a, info = action_info(planner, s; tree_in_info=true,
                          α=α, β=β, γ=γ, schedule=0.0)
    tree = info[:tree]
    output = (tree.costs, tree.weights, info)
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
