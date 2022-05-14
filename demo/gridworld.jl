using BSON, Dates, Distributions, FileIO, Plots, Printf, StaticArrays

using Crux, Flux, POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies

using ParallelTreeSampling

using ParallelTreeSampling

include("utils.jl")

unicodeplots()

# Basic MDP.
tprob = 0.7
Random.seed!(0)
randcosts = Dict(POMDPGym.GWPos(i,j) => rand() for i = 1:10, j=1:10)
# zerocosts = Dict(POMDPGym.GWPos(i,j) => 0.0 for i = 1:10, j=1:10)
mdp = GridWorldMDP(costs=randcosts, cost_penalty=0.1, tprob=tprob)

# Learn a policy that solves it.
# policy = DiscreteNetwork(Chain(x -> (x .- 5f0) ./ 5f0, Dense(2, 32, relu), Dense(32, 4)), [:up, :down, :left, :right])
# policy = solve(DQN(π=policy, S=state_space(mdp), N=100000, ΔN=4, buffer_size=10000, log=(;period=5000)), mdp)
# atable = Dict(s => action(policy, [s...]) for s in states(mdp.g))
# BSON.@save "demo/gridworld_policy_table.bson" atable

atable = BSON.load("demo/policies/gridworld_policy_table.bson")[:atable]

# Define the adversarial mdp.
adv_rewards = deepcopy(randcosts)
# adv_rewards = deepcopy(zerocosts)
for (k,v) in mdp.g.rewards
    if v < 0
        adv_rewards[k] += -10*v
    end
end

amdp = GridWorldMDP(rewards=adv_rewards, tprob=1., discount=1., terminate_from=mdp.g.terminate_from)

# Define action probability for the adv_mdp.
action_probability(mdp, s, a) = (a == atable[s][1]) ? tprob : ((1. - tprob) / (length(actions(mdp)) - 1.))

function distribution(mdp::Union{POMDP,MDP}, s)
    xs = POMDPs.actions(mdp, s)
    ps = [action_probability(mdp, s, x) for x in xs]
    ps ./= sum(ps)
    px = GenericDiscreteNonParametric(xs, ps)
    return px
end

fixed_s = rand(initialstate(amdp))

base_N = 1_000_000
# Parameters for each α.
params = Dict(
    1e-1 => (N=100_000, c=0.0, vloss=0.0, β=0.1, γ=0.3),  # Default.
    1e-2 => (N=100_000, c=0.0, vloss=0.0, β=0.1, γ=0.3),
    1e-3 => (N=100_000, c=0.0, vloss=0.0, β=0.1, γ=0.3),
    1e-4 => (N=100_000, c=0.0, vloss=0.0, β=0.1, γ=0.3),
    1e-5 => (N=100_000, c=0.0, vloss=0.0, β=0.1, γ=0.3),
)
a_selection = :adaptive
save_output = false
is_baseline = false
num_trials = 1

path = "data"

# alpha_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
alpha_list = [1e-3, 1e-4]
date_str = Dates.format(Dates.now(), "yyyy-mm-dd")
time_str = Dates.format(Dates.now(), "HHMM")


function baseline(trial=0)
    mc_out = run_mc(amdp, fixed_s, distribution; N=base_N)
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
    αp = haskey(params, α) ? params[α] : params[1e-1]
    N, c, vloss, β, γ = αp[:N], αp[:c], αp[:vloss], αp[:β], αp[:γ]
    mcts_out, planner = run_mcts(
        amdp, fixed_s, distribution, a_selection; N=N, c=c, vloss=vloss, α=α, β=β, γ=γ)
    @assert length(mcts_out[1]) == N
    mcts_info = mcts_out[3]
    if save_output
        filename = string("gwl_is_$(a_selection)_$(time_str)_$(N)",
                          (trial > 0 ? string("_", trial) : "") ,".jld2")
        save(joinpath(path, filename),
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
        Random.seed!(trial * length(alpha_list) + idx)
        if !is_baseline
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


# metrics = []
# times = []
# num_trials = 1
# for trial in 1:num_trials
#     Random.seed!(trial)
#     if is_baseline
#         push!(metrics, baseline(trial))
#     else
#         metric_d, search_t = mcts(trial)
#         push!(metrics, metric_d)
#         push!(times, search_t)
#     end
# end

# if is_baseline
#     println("Baseline metrics")
# else
#     println("TIS metrics: N=$(N), c=$(c), α=$(α), β=$(β), γ=$(γ)")
# end

# for (idx, metric) in enumerate(metrics)
#     println("Run $(idx)")
#     for alpha in alpha_list
#         m_d = metric[alpha]
#         println(string("  [Alpha=$(alpha)] Mean: $(m_d[:mean]), ",
#                        "VaR: $(m_d[:var]), CVaR: $(m_d[:cvar]), Worst: $(m_d[:worst])"))
#     end
# end
# println("Times: $(times)")
