using FileIO
using PyPlot

using ParallelTreeSampling

include("utils.jl")


# Parameters to be set for plotting.
mc_data = ["data/gridworld_baseline_10000000_$(trial).jld2" for trial in 1:5]
is_data = ["data/gridworld_mcts_100000_$(trial).jld2" for trial in 1:5]
delta_n_mc = 100_000; delta_n_is = 10_000;
plot_hist = false; plot_est = true;

# Constants to be updated as needed.
samples_key = "risks:"; weights_key = "weights:";

# Read the jld2 files.
function load_jld2(jld2_filename)
    d = load(jld2_filename)
    @assert haskey(d, samples_key)
    return (haskey(d, weights_key) ?
        (d[samples_key], d[weights_key]) : d[samples_key])
end
n_mc_samples = [load_jld2(mc_datum) for mc_datum in mc_data]
n_is_samples = [load_jld2(is_datum) for is_datum in is_data]


# Returns a lambda for computing metrics used for plotting.
#
# Valid metric types include :mean, :var, :cvar, :worst.
function estimate_fn(metric_type, alpha, with_weights=false)
    if with_weights
        fn = function(samples, weights)
            metrics = eval_metrics(samples; weights=exp.(weights), alpha=alpha)
            return getproperty(metrics, metric_type)
        end
    else
        fn = function(samples)
            metrics = eval_metrics(samples; weights=nothing, alpha=alpha)
            return getproperty(metrics, metric_type)
        end
    end
    return fn
end


# Plot convergence graphs.
if plot_est
    figure(figsize=(9.0, 6.0))
    if !isempty(n_mc_samples)
        plot_estimates(n_mc_samples, estimate_fn(:var, 1e-3),
                       delta_n_mc, "MC-VaR-1e-3")
        plot_estimates(n_mc_samples, estimate_fn(:cvar, 1e-3),
                       delta_n_mc, "MC-CVaR-1e-3")
        plot_estimates(n_mc_samples, estimate_fn(:var, 1e-4),
                       delta_n_mc, "MC-VaR-1e-4")
        plot_estimates(n_mc_samples, estimate_fn(:cvar, 1e-4),
                       delta_n_mc, "MC-CVaR-1e-4")
    end
    if !isempty(n_is_samples)
        plot_estimates(n_is_samples, estimate_fn(:var, 1e-3, true),
                       delta_n_is, "IS-VaR-1e-3")
        plot_estimates(n_is_samples, estimate_fn(:cvar, 1e-3, true),
                       delta_n_is, "IS-CVaR-1e-3")
        plot_estimates(n_is_samples, estimate_fn(:var, 1e-4, true),
                       delta_n_is, "IS-VaR-1e-4")
        plot_estimates(n_is_samples, estimate_fn(:cvar, 1e-4, true),
                       delta_n_is, "IS-CVaR-1e-4")
    end
    xlabel("no. samples"); ylabel("estimates"); legend();
end

# Plot histograms.
if plot_hist
    figure(figsize=(9.0, 6.0))
    if !isempty(n_mc_samples)
        samples = first(n_mc_samples)
        plot_histogram(samples, bins=50, label="MC")
    end
    if !isempty(n_is_samples)
        samples, weights = first(n_is_samples)
        plot_histogram(samples, bins=50, label="IS-uw")
        plot_histogram(samples, weights, bins=50, label="IS-w")
    end
    ylim(bottom=0.0); xlabel("costs"); ylabel("density"); legend();
end
