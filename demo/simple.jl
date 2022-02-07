using Distributions
using FileIO
using POMDPGym
using POMDPs
using Random

include("utils.jl")


# Define SimpleMDP.
struct SimpleMDP <: MDP{Float64, Float64} end

POMDPs.initialstate(mdp::SimpleMDP) = MersenneTwister()
POMDPs.actions(mdp::SimpleMDP) = DiscreteNonParametric([1.0, 2.0], [0.5, 0.5])
POMDPs.isterminal(mdp::SimpleMDP, s) = s > 5
POMDPs.discount(mdp::SimpleMDP) = 1.0

function POMDPs.gen(mdp::SimpleMDP, s, a, rng::AbstractRNG=Random.GLOBAL_RNG; kwargs...)
    sp = s + a
    r = sp > 2 ? 1 : 2
    return (sp=sp, r=r)
end

mdp = SimpleMDP()

fixed_s = rand(initialstate(mdp))

N = 1000
c = 0.3
vloss = 0.0
α = 0.1
β = 1.0
γ = 0.0

path = "/home/kykim/dev/sisl/ParallelTreeSampling/data"

tree_mdp = create_tree_amdp(mdp, actions; reduction="sum")

baseline_out = run_baseline(mdp, fixed_s, actions; N)
mcts_out, planner = run_mcts(tree_mdp, fixed_s; N=1000, c=0.3, vloss=vloss, α=α, β=β, γ=γ)

save(joinpath(path, "simple_baseline_$(N).jld2"),
     Dict("risks:" => baseline_out[1], "states:" => baseline_out[2]))
save(joinpath(path, "simple_mcts_$(N).jld2"),
     Dict("risks:" => mcts_out[1], "states:" => mcts_out[2], "IS_weights:" => mcts_out[3], "tree:" => mcts_out[4]))


alpha_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5];

println("Baseline metrics")
for alpha in alpha_list
    m = eval_metrics(baseline_out[1]; alpha)
    println("[Alpha=$(alpha)] Mean: $(m.mean), VaR: $(m.var), CVaR: $(m.cvar), Worst: $(m.worst)")
end

println()
println("TIS metrics: N=$(N), c=$(c), α=$(α), β=$(β), γ=$(γ)")
for alpha in alpha_list
    m = eval_metrics(mcts_out[1]; weights=exp.(mcts_out[3]), alpha=alpha)
    println("[Alpha=$(alpha)] Mean: $(m.mean), VaR: $(m.var), CVaR: $(m.cvar), Worst: $(m.worst)")
end
