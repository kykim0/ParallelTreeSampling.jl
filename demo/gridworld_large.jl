using BSON, Dates, Distributions, FileIO, PyPlot, Printf, StaticArrays
using Crux, Flux, POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies
using ParallelTreeSampling

include("utils.jl")

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

_p_0 = 0.90; _p_1 = (1.0 - _p_0) / 4
px = GenericDiscreteNonParametric(
    [GWPos(0,0), GWPos(3,0), GWPos(0,3), GWPos(-3,0), GWPos(0,-3)],
    [_p_0, _p_1, _p_1, _p_1, _p_1])
function cost_fn(rmdp::RMDP, s, sp)
    amdp = rmdp.amdp
    cost = get(amdp.costs, s, 0)
    reward = POMDPs.reward(amdp, s)
    return reward < 0.0 ? -reward : amdp.cost_penalty * cost * 1.0
end
rmdp = RMDP(amdp=mdp, π=FunctionPolicy(policy_fn), cost_fn=cost_fn, disturbance_type=:noise)

POMDPs.actions(mdp::RMDP) = px
POMDPs.actionindex(mdp::RMDP, x) = findfirst(px.support .== x)


fixed_s = GWPos(10,10)

base_N = 100_000
# Parameters for each α.
params = Dict(
    1e-1 => (N=100_000, c=3.0, vloss=0.0, α=1e-1, min_s=5.0,
             mix_w_fn=linear_decay_schedule(1.0, 0.95, 50_000)),
    5e-2 => (N=100_000, c=3.0, vloss=0.0, α=5e-2, min_s=5.0,
             mix_w_fn=linear_decay_schedule(1.0, 0.95, 50_000)),
    1e-2 => (N=100_000, c=3.0, vloss=0.0, α=1e-2, min_s=5.0,
             mix_w_fn=linear_decay_schedule(1.0, 0.95, 50_000)),
    1e-3 => (N=10_000, c=5.0, vloss=0.0, α=1e-3, min_s=5.0,
             mix_w_fn=(n) -> 0.5),
)
a_selection = :var_sigmoid
rollout_s = :nominal
save_output = false
is_baseline = false
plot_est = true
num_trials = 3

path = "data"

alpha_list = [1e-2]
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
    N, c, vloss, target_α, mix_w_fn, min_s = αp[:N], αp[:c], αp[:vloss], αp[:α], αp[:mix_w_fn], αp[:min_s]
    mcts_out, planner = run_mcts(
        rmdp, fixed_s, nominal_distrib_fn, a_selection, rollout_s, mix_w_fn, min_s;
        N=N, c=c, vloss=vloss, α=target_α)
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


alpha_raw = Dict(alpha => [] for alpha in alpha_list)
alpha_metrics = Dict(alpha => [] for alpha in alpha_list)
alpha_times = Dict(alpha => [] for alpha in alpha_list)
for trial in 1:num_trials
    Random.seed!(trial * 2)
    costs = is_baseline ? baseline(trial) : nothing
    local weights = nothing
    for (idx, alpha) in enumerate(alpha_list)
        if !is_baseline
            Random.seed!(trial * length(alpha_list) + idx)
            costs, weights, search_t = mcts(alpha, trial)
            weights = exp.(weights)
            push!(alpha_times[alpha], search_t)
        end
        raw_data = isnothing(weights) ? costs : (costs, weights)
        push!(alpha_raw[alpha], raw_data)
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

if plot_est
    PyPlot.plt.figure(figsize=(9.0, 6.0))
    delta_n = 100; log_scale = true
    for alpha in alpha_list
        alpha_str = @sprintf("%.1e", alpha)
        n_samples = alpha_raw[alpha]

        v_ret = plot_estimates(n_samples, estimate_fn(:var, alpha), delta_n,
                               "VaR-$(alpha_str)", log_scale)
        cv_ret = plot_estimates(n_samples, estimate_fn(:cvar, alpha), delta_n,
                                "CVaR-$(alpha_str)", log_scale)

        v_x, v_min_y, v_mid_y, v_max_y = v_ret
        cv_x, cv_min_y, cv_mid_y, cv_max_y = cv_ret
    end
    xlabel("no. samples"); ylabel("estimates"); legend();
end
