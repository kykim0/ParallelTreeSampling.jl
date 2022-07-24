using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux
using FileIO

# Basic MDP
mdp = InvertedPendulumMDP(λcost=0, include_time_in_state=true)

# Learn a policy that solves it
policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))), [0f0]), 
                     ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))))
policy = solve(PPO(π=policy, S=state_space(mdp), N=2000, ΔN=400), mdp)

# Define the disturbance distribution based on a normal distribution
xnom = Normal(0f0, 0.5f0)
xs = Float32[-2., -0.5, 0, 0.5, 2.]
ps = exp.([logpdf(xnom, x) for x in xs])
ps ./= sum(ps)
px = DiscreteNonParametric(xs, ps)

# Redefine disturbance to find action space
POMDPGym.disturbances(mdp::AdditiveAdversarialMDP) = support(mdp.x_distribution)
POMDPGym.disturbanceindex(mdp::AdditiveAdversarialMDP, x) = findfirst(support(mdp.x_distribution) .== x)

prior(mdp::AdditiveAdversarialMDP) = probs(mdp.x_distribution)

# Construct the adversarial MDP to get access to a transition function like gen(mdp, s, a, x)
amdp = AdditiveAdversarialMDP(mdp, px)

function eval_cost(x)
    x_inv = 1 / (abs(x - mdp.failure_thresh) + 1e-3)
    cost = min(x_inv, 10.0)
    cost = cost/10.0
    return cost
end

# Construct the risk estimation mdp where actions are disturbances
rmdp = RMDP(amdp, policy, (m, s) -> eval_cost(s[1]))

# N = 10000
N = 100_000
c = 0.3

# BASELINE

# samps = [maximum(collect(simulate(HistoryRecorder(), rmdp, FunctionPolicy((s) -> rand(px)))[:r])) for _ in 1:N]

# save("/Users/kykim/Desktop/inverted_pendulum_baseline_$(N).jld2", Dict("risks:" => samps, "states:" => [], "IS_weights:" => []))
# save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/inverted_pendulum_baseline_$(N).jld2", Dict("risks:" => samps, "states:" => [], "IS_weights:" => []))

# MCTS

tree_mdp = return TreeMDP(rmdp, 1.0, [], [], px)
solver = PISSolver(; depth=100, 
                   estimate_value=rollout,  # Required.
                   exploration_constant=c,
                   n_iterations=N,
                   enable_state_pw=false,   # Required.
                   show_progress=true,
                   tree_in_info=true,
                   virtual_loss=vloss);
planner = solve(solver, tree_mdp);

a, w, info = action_info(planner, TreeState(rand(initialstate(rmdp))), tree_in_info=true)

N_dist = length(xs)

# save("/Users/kykim/Desktop/inverted_pendulum_mcts_IS_$(N).jld2", Dict("risks:" => planner.mdp.costs, "states:" => [], "IS_weights:" => planner.mdp.IS_weights, "tree:" => info[:tree]))
# save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/inverted_pendulum_mcts_IS_$(N).jld2", Dict("risks:" => planner.mdp.costs, "states:" => [], "IS_weights:" => planner.mdp.IS_weights, "tree:" => info[:tree]))




#=
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux
using FileIO

include("tree_mdp.jl")


# Basic MDP.
mdp = InvertedPendulumMDP(λcost=0, include_time_in_state=true)

# Learn a policy that solves it.
policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))), [0f0]), 
                     ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))))
policy = solve(PPO(π=policy, S=state_space(mdp), N=2000, ΔN=400), mdp)

# Define the disturbance distribution based on a normal distribution.
xnom = Normal(0f0, 0.5f0)
xs = Float32[-2., -0.5, 0, 0.5, 2.]
ps = exp.([logpdf(xnom, x) for x in xs])
ps ./= sum(ps)
px = DiscreteNonParametric(xs, ps)

# Re-define disturbance to find action space.
POMDPGym.disturbances(mdp::AdditiveAdversarialMDP) = support(mdp.x_distribution)
POMDPGym.disturbanceindex(mdp::AdditiveAdversarialMDP, x) = findfirst(support(mdp.x_distribution) .== x)

prior(mdp::AdditiveAdversarialMDP) = probs(mdp.x_distribution)

# Construct the adversarial MDP to get access to a transition function like gen(mdp, s, a, x).
# amdp = AdditiveAdversarialMDP(mdp, px)

# Construct the risk estimation mdp where actions are disturbances.
function eval_cost(x)
    x_inv = 1 / (abs(x - mdp.failure_thresh) + 1e-3)
    cost = min(x_inv, 10.0)
    cost = cost/10.0
    return cost
end

amdp = InvertedPendulumMDP(rewards=eval_cost, tprob=1., discount=1.)
# mdp = InvertedPendulumMDP(λcost=0, include_time_in_state=true)

# rmdp = RMDP(amdp, policy, (m, s) -> eval_cost(s[1]))
basedir = "/home/kykim/Desktop"

# Baseline.
samples = [maximum(collect(simulate(HistoryRecorder(), amdp, FunctionPolicy((s) -> rand(px)))[:r])) for _ in 1:N]
# baseline_costs = [sum(collect(simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> rand(disturbance(mdp, s))), fixed_s)[:r])) for _ in 1:N]
# baseline_output = (baseline_costs, [])

save(joinpath(basedir, "inverted_pendulum_baseline_$(N).jld2"),
     Dict("risks:" => samples, "states:" => [], "IS_weights:" => []))

# Tree sampling.
N = 10_000
c = 0.3
vloss = 3.0

tree_mdp = TreeMDP(amdp, 1.0, [], [], (m, s) -> px)
solver = PISSolver(; depth=100, 
                      estimate_value=rollout,  # Required.
                      exploration_constant=c,
                      n_iterations=N,
                      enable_state_pw=false,   # Required.
                      show_progress=true,
                      tree_in_info=true,
                      virtual_loss=vloss);
planner = solve(solver, tree_mdp);
a, w, info = action_info(planner, TreeState(rand(initialstate(rmdp))), tree_in_info=true)
N_dist = length(xs)

save(joinpath(basedir, "inverted_pendulum_is_$(N).jld2"),
     Dict("risks:" => planner.mdp.costs, "states:" => [], "IS_weights:" => planner.mdp.IS_weights, "tree:" => info[:tree]))
=#
