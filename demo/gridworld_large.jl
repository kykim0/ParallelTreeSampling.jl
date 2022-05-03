using BSON, Dates, Distributions, FileIO, Plots, StaticArrays

using Crux, Flux, POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies

using ParallelTreeSampling

include("utils.jl")

unicodeplots()

# Create a larger GridWorldMDP.
# Use a different set of rewards, especially with a relatively large negative
# reward far out from the center of the grid.
# Note that the default is (4,3)=>-10.0, (4,6)=>-5.0, (9,3)=>10.0, (8,8)=>3.0.
tprob = 0.7
rewards = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0,
               GWPos(8,8)=>3.0, GWPos(16,18)=>-30.0)
# Add small costs for other grids.
Random.seed!(0)
rand_costs = Dict(GWPos(i,j) => rand()
                  for i in 1:20, j in 1:20 if !haskey(rewards, GWPos(i,j)))
mdp = GridWorldMDP(size=(20,20), rewards=rewards, costs=rand_costs, cost_penalty=1.0)

# Learn a policy that solves it.
function train_dqn(mdp::GridWorldMDP)
    N = 5_000_000
    policy = DiscreteNetwork(
        Chain(x -> (x .- 10f0) ./ 10f0, Dense(2, 32, relu), Dense(32, 4)),
        [:up, :down, :left, :right])
    policy = solve(DQN(π=policy, S=state_space(mdp), N=N, ΔN=4,
                       buffer_size=10_000, log=(;period=round(Int, N/10))), mdp)
    atable = Dict(s=>action(policy, [s...]) for s in states(mdp.g))
    BSON.@save "demo/policies/gridworld_large_policy_table.bson" atable
end
# train_dqn(mdp)

atable = BSON.load("demo/policies/gridworld_large_policy_table.bson")[:atable]
function policy_fn(s)
    if 1 <= s[1] <= mdp.g.size[1] && 1 <= s[2] <= mdp.g.size[2]
        return atable[s][1]
    end
    xs = POMDPs.actions(mdp, s)
    return xs[rand(1:end)]
end

px = GenericDiscreteNonParametric(
    [GWPos(0,0),
     GWPos(1,0), GWPos(0,1), GWPos(-1,0), GWPos(0,-1),
     GWPos(2,0), GWPos(0,2), GWPos(-2,0), GWPos(0,-2)],
    [0.600,
     0.075, 0.075, 0.075, 0.075,
     0.025, 0.025, 0.025, 0.025])
function cost_fn(rmdp::RMDP, s, sp)
    amdp = rmdp.amdp
    cost = get(amdp.costs, s, 0)
    reward = POMDPs.reward(amdp, s)
    return reward < 0.0 ? -reward : amdp.cost_penalty * cost
end
rmdp = RMDP(amdp=mdp, π=FunctionPolicy(policy_fn), cost_fn=cost_fn, disturbance_type=:noise)

POMDPs.actions(mdp::RMDP) = px
POMDPs.actionindex(mdp::RMDP, x) = findfirst(px.support .== x)


fixed_s = GWPos(10,10)

N = 100_000
base_N = 1_000_000
c = 0.0
α = 0.001; β = 0.1; γ = 0.3
vloss = 0.0
a_selection = :ucb
save_output = true
is_baseline = false

path = "data"

alpha_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
date_str = Dates.format(Dates.now(), "yyyy-mm-dd")
time_str = Dates.format(Dates.now(), "HHMM")

function baseline(trial=0)
    mc_out = run_mc(rmdp, fixed_s, px; N=base_N)
    @assert length(mc_out[1]) == base_N
    if save_output
        filename = string("gwl_mc_$(time_str)_$(base_N)",
                          (trial > 0 ? string("_", trial) : "") ,".jld2")
        base_dir = joinpath(path, date_str)
        mkpath(base_dir)
        save(joinpath(base_dir, filename), Dict("risks:" => mc_out[1]))
    end

    d = Dict([(alpha, Dict()) for alpha in alpha_list])
    for alpha in alpha_list
        m = eval_metrics(mc_out[1]; alpha)
        d[alpha][:mean] = round(m.mean, digits=3);
        d[alpha][:var] = round(m.var, digits=3); d[alpha][:cvar] = round(m.cvar, digits=3);
        d[alpha][:worst] = round(m.worst, digits=3);
    end
    return d
end


function mcts(trial=0)
    nominal_distrib_fn = (mdp, s) -> px
    mcts_out, planner = run_mcts(rmdp, fixed_s, nominal_distrib_fn, a_selection;
                                 N=N, c=c, vloss=vloss, α=α, β=β, γ=γ)
    @assert length(mcts_out[1]) == N
    mcts_info = mcts_out[4]
    if save_output
        filename = string("gwl_is_$(a_selection)_$(time_str)_$(N)",
                          (trial > 0 ? string("_", trial) : "") ,".jld2")
        base_dir = joinpath(path, date_str)
        mkpath(base_dir)
        save(joinpath(base_dir, filename),
             Dict("risks:" => mcts_out[1], "states:" => mcts_out[2],
                  "weights:" => mcts_out[3], "tree:" => mcts_info[:tree]))
    end
    search_t = round(mcts_info[:search_time_s]; digits=3)

    d = Dict([(alpha, Dict()) for alpha in alpha_list])
    for alpha in alpha_list
        m = eval_metrics(mcts_out[1]; weights=exp.(mcts_out[3]), alpha=alpha)
        d[alpha][:mean] = round(m.mean, digits=3);
        d[alpha][:var] = round(m.var, digits=3); d[alpha][:cvar] = round(m.cvar, digits=3);
        d[alpha][:worst] = round(m.worst, digits=3);
    end
    return d, search_t
end


metrics = []
times = []
num_trials = 3
for trial in 1:num_trials
    Random.seed!(trial)
    if is_baseline
        push!(metrics, baseline(trial))
    else
        metric_d, search_t = mcts(trial)
        push!(metrics, metric_d)
        push!(times, search_t)
    end
end

if is_baseline
    println("Baseline metrics")
else
    println("TIS metrics: N=$(N), c=$(c), α=$(α), β=$(β), γ=$(γ)")
end

for (idx, metric) in enumerate(metrics)
    println("Run $(idx)")
    for alpha in alpha_list
        m_d = metric[alpha]
        println(string("  [Alpha=$(alpha)] Mean: $(m_d[:mean]), ",
                       "VaR: $(m_d[:var]), CVaR: $(m_d[:cvar]), Worst: $(m_d[:worst])"))
    end
end
println("Times: $(times)")
