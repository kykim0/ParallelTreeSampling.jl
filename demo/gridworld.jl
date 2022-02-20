using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using Crux, Flux, BSON, StaticArrays, Random
using MCTS
using FileIO
using Plots
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

# Define the adversarial mdp
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


# Generic Discrete NonParametric with symbol support.
struct GenericDiscreteNonParametric
    g_support::Any
    pm::DiscreteNonParametric
end

GenericDiscreteNonParametric(vs::T, ps::Ps) where {
        T<:Any,P<:Real,Ps<:AbstractVector{P}} = GenericDiscreteNonParametric([v for v in vs], DiscreteNonParametric([i for i=1:length(vs)], ps))

Distributions.support(d::GenericDiscreteNonParametric) = d.g_support

Distributions.probs(d::GenericDiscreteNonParametric)  = d.pm.p

function Base.rand(rng::AbstractRNG, d::GenericDiscreteNonParametric)
    x = support(d)
    p = probs(d)
    n = length(p)
    draw = rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i +=1]
    end
    return x[i]
end

function Distributions.pdf(d::GenericDiscreteNonParametric, x::Any)
    s = support(d)
    idx = findfirst(isequal(x), s)
    ps = probs(d)
    if idx <= length(ps) && s[idx] == x
        return ps[idx]
    else
        return zero(eltype(ps))
    end
end
Distributions.logpdf(d::GenericDiscreteNonParametric, x::Any) = log(pdf(d, x))

function disturbance(m::typeof(amdp), s)
    xs = POMDPs.actions(m, s)
    ps = [action_probability(m, s, x) for x in xs]
    ps ./= sum(ps)
    px = GenericDiscreteNonParametric(xs, ps)
    return px
end

fixed_s = rand(initialstate(amdp))

N = 100_000
base_N = 10_000_000
c = 0.0
α = 0.001
β = 0.1
γ = 0.3
vloss = 0.0
is_baseline = false

path = "data"

tree_mdp = create_tree_amdp(amdp, disturbance; reduction="sum")

alpha_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


function baseline()
    baseline_out = run_baseline(amdp, fixed_s, disturbance; N=base_N)
    save(joinpath(path, "gridworld_baseline_$(N).jld2"),
         Dict("risks:" => baseline_out[1], "states:" => baseline_out[2]))

    d = Dict([(alpha, Dict()) for alpha in alpha_list])
    for alpha in alpha_list
        m = eval_metrics(baseline_out[1]; alpha)
        d[alpha][:mean] = m.mean; d[alpha][:var] = m.var; d[alpha][:cvar] = m.cvar; d[alpha][:worst] = m.worst;
    end
    return d
end


function mcts()
    mcts_out, planner = run_mcts(tree_mdp, fixed_s; N=N, c=c, vloss=vloss, α=α, β=β, γ=γ)
    mcts_info = mcts_out[4]
    save(joinpath(path, "gridworld_mcts_$(N).jld2"),
         Dict("risks:" => mcts_out[1], "states:" => mcts_out[2], "IS_weights:" => mcts_out[3], "tree:" => mcts_info[:tree]))
    search_t = round(mcts_info[:search_time_s]; digits=3)

    d = Dict([(alpha, Dict()) for alpha in alpha_list])
    for alpha in alpha_list
        m = eval_metrics(mcts_out[1]; weights=exp.(mcts_out[3]), alpha=alpha)
        d[alpha][:mean] = m.mean; d[alpha][:var] = m.var; d[alpha][:cvar] = m.cvar; d[alpha][:worst] = m.worst;
    end
    return d, search_t
end


metrics = []
times = []
num_iter = 3
for idx in 1:num_iter
    Random.seed!(idx-1)
    if is_baseline
        push!(metrics, baseline())
    else
        metric_d, search_t = mcts()
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
        println("  [Alpha=$(alpha)] Mean: $(m_d[:mean]), VaR: $(m_d[:var]), CVaR: $(m_d[:cvar]), Worst: $(m_d[:worst])")
    end
end
println("Times: $(times)")
