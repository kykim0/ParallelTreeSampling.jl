using BSON, CSV, DataFrames, Dates, Distributions, FileIO, Plots, Printf, StaticArrays
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
function mcts(α::Float64; kwargs...)
    nominal_distrib_fn = (mdp, s) -> px
    params = Dict(kwargs)
    N = params[:N]; c = params[:c]; vloss = params[:vloss]
    β = params[:β]; γ = params[:γ]; a_select = params[:a_select]
    mcts_out, planner = run_mcts(rmdp, fixed_s, nominal_distrib_fn, a_select;
                                 N=N, c=c, vloss=vloss, α=α, β=β, γ=γ,
                                 show_progress=false)
    @assert length(mcts_out[1]) == N
    mcts_info = mcts_out[3]
    return mcts_out[1], mcts_out[2], mcts_info[:search_time_s]
end

# Get ground trutch estimates
mc_samps = load("data/2022-05-08/gwl_mc_1712_1000000_1.jld2")["risks:"]
mc_weights = ones(length(mc_samps))

# Parameters to search over.
α_vec = [1e-2, 1e-3, 1e-4, 1e-5]
α_vec = [1e-3]
N_vec = [100_000]
N_vec = [1000]
c_vec = [0.0]
β_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
γ_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
a_selection_vec = [:expected_cost, :mixture, :adaptive]

output_dir = "/home/kykim/Desktop"  # Change as needed.
mkpath(output_dir)
df = DataFrame(alpha=Float64[], N=Integer[], explore_const=Float64[],
               beta=Float64[], gamma=Float64[], a_selection=String[],
               var_err=Float64[], cvar_err=Float64[], worst=Float64[])

all_params = [p for p in Iterators.product(α_vec, N_vec, c_vec, β_vec, γ_vec, a_selection_vec)]
for (idx, params) in enumerate(all_params)
    println("Running ", idx, " / ", length(all_params), " at ",
            Dates.format(Dates.now(), "HH:MM"))
    α = params[1]; N = params[2]; c = params[3]
    β = params[4]; γ = params[5]; a_select = params[6]; vloss=0.0

    samples, weights, search_t = mcts(α; N=N, c=c, vloss=vloss, β=β, γ=γ, a_select=a_select)

    mc_metrics = IWRiskMetrics(mc_samps, mc_weights, α)
    is_metrics = IWRiskMetrics(samples, exp.(weights), α)
    var_err = abs(mc_metrics.var - is_metrics.var) / mc_metrics.var
    cvar_err = abs(mc_metrics.cvar - is_metrics.cvar) / mc_metrics.cvar
    worst = is_metrics.worst

    global df
    push!(df, [α, N, c, β, γ, string(a_select),
               var_err, cvar_err, worst])
    if idx % 20 == 0
        CSV.write(joinpath(output_dir, string("gw_tune_$(idx).csv")), df)
    end
end
CSV.write(joinpath(output_dir, "gw_tune.csv"), df)
