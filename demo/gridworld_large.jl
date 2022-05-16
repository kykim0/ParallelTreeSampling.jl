using BSON, Dates, Distributions, FileIO, Plots, Printf, StaticArrays
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

base_N = 10_000_000
# Parameters for each α.
params = Dict(
    1e-1 => (N=100_000, c=0.0, vloss=0.0, α=1e-1, β=0.1, γ=0.3),  # Default.
    1e-2 => (N=100_000, c=0.0, vloss=0.0, α=1e-2, β=0.1, γ=0.3),
    1e-3 => (N=100_000, c=0.0, vloss=0.0, α=1e-3, β=0.1, γ=0.3),
    1e-4 => (N=100_000, c=0.0, vloss=0.0, α=1e-4, β=0.1, γ=0.3),
    1e-5 => (N=100_000, c=0.0, vloss=0.0, α=1e-5, β=0.1, γ=0.3),
)
a_selection = :adaptive
save_output = false
is_baseline = false
num_trials = 1

path = "data"

# alpha_list = [1e-2, 1e-3, 1e-4, 1e-5]
alpha_list = [1e-3, 1e-4]
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
    return mc_out[1]
end


function mcts(α::Float64, trial=0)
    nominal_distrib_fn = (mdp, s) -> px
    αp = haskey(params, α) ? params[α] : params[1e-1]
    N, c, vloss, β, γ, target_α = αp[:N], αp[:c], αp[:vloss], αp[:β], αp[:γ], αp[:α]
    mcts_out, planner = run_mcts(rmdp, fixed_s, nominal_distrib_fn, a_selection;
                                 N=N, c=c, vloss=vloss, α=target_α, β=β, γ=γ)
    @assert length(mcts_out[1]) == N
    mcts_info = mcts_out[3]
    if save_output
        filename = string("gwl_is_$(a_selection)_$(time_str)_$(N)",
                          (trial > 0 ? string("_", trial) : "") ,".jld2")
        base_dir = joinpath(path, date_str)
        mkpath(base_dir)
        save(joinpath(base_dir, filename),
             Dict("risks:" => mcts_out[1], "weights:" => mcts_out[2],
                  "tree:" => mcts_info[:tree]))
    end
    return mcts_out[1], mcts_out[2], mcts_info[:search_time_s]
end


alpha_metrics = Dict(alpha => [] for alpha in alpha_list)
alpha_times = Dict(alpha => [] for alpha in alpha_list)
for trial in 1:num_trials
    Random.seed!(trial)
    costs = is_baseline ? baseline(trial) : nothing
    weights = nothing
    for (idx, alpha) in enumerate(alpha_list)
        if !is_baseline
            Random.seed!(trial * length(alpha_list) + idx)
            costs, weights, search_t = mcts(alpha, trial)
            weights = exp.(weights)
            push!(alpha_times[alpha], search_t)
        end
        m = eval_metrics(costs; weights=weights, alpha=alpha)
        m_tuple = (mean=m.mean, var=m.var, cvar=m.cvar, worst=m.worst)
        push!(alpha_metrics[alpha], m_tuple)
    end
end

is_baseline ? println("Baseline metrics") : println("Tree sampling metrics")
for alpha in alpha_list
    m_symbols = [:mean, :var, :cvar, :worst]
    m_dict = Dict(m_symbol => [] for m_symbol in m_symbols)
    for m_tuple in alpha_metrics[alpha]
        for m_symbol in m_symbols
            push!(m_dict[m_symbol], m_tuple[m_symbol])
        end
    end
    print("[Alpha=$(alpha)]")
    for m_symbol in m_symbols
        m_vec = m_dict[m_symbol]
        m_mean, m_max = mean(m_vec), maximum(m_vec)
        print("\t$(m_symbol): $(@sprintf("%.3f", m_mean))±$(@sprintf("%.3f", m_max-m_mean))")
    end
    search_ts = alpha_times[alpha]
    if !isempty(search_ts)
        t_mean, t_max = mean(search_ts), maximum(search_ts)
        print("\ttime: $(@sprintf("%.3f", t_mean))±$(@sprintf("%.3f", t_max-t_mean))")
    end
    println()
end
