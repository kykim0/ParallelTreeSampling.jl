using FileIO
using Printf
using PyPlot

using ParallelTreeSampling

include("utils.jl")


# Parameters to be set for plotting.
mc_base_dir = "data/2022-05-03"; is_base_dir = "data/2022-05-03"
mc_data = [joinpath(mc_base_dir, "gwl_mc_1636_1000000_$(trial).jld2") for trial in 1:3]
# mc_data = []
is_data = [joinpath(is_base_dir, "gwl_is_ucb_1615_100000_$(trial).jld2") for trial in 1:3]
# is_data = []
max_num_mc = -1; max_num_is = -1
delta_n_mc = 100; delta_n_is = 100
plot_hist = true; plot_est = true; log_scale = true
output_dir = homedir()

# Constants to be updated as needed.
samples_key = "risks:"; weights_key = "weights:"

# Read the jld2 files.
function load_jld2(jld2_filename)
    d = load(jld2_filename)
    @assert haskey(d, samples_key)
    return (haskey(d, weights_key) ?
        (d[samples_key], d[weights_key]) : d[samples_key])
end
n_mc_samples = [load_jld2(mc_datum)[1:(max_num_mc < 0 ? end : max_num_mc)]
                for mc_datum in mc_data]
n_is_samples = [load_jld2(is_datum)[1:(max_num_is < 0 ? end : max_num_is)]
                for is_datum in is_data]


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
alphas = [1e-4]
if plot_est
    PyPlot.plt.figure(figsize=(9.0, 6.0))
    if !isempty(n_mc_samples)
        for alpha in alphas
            alpha_str = @sprintf("%.1e", alpha)
            plot_estimates(n_mc_samples, estimate_fn(:var, alpha),
                           delta_n_mc, "MC-VaR-$(alpha_str)", log_scale)
            plot_estimates(n_mc_samples, estimate_fn(:cvar, alpha),
                           delta_n_mc, "MC-CVaR-$(alpha_str)", log_scale)
        end
    end
    if !isempty(n_is_samples)
        for alpha in alphas
            alpha_str = @sprintf("%.1e", alpha)
            plot_estimates(n_is_samples, estimate_fn(:var, alpha, true),
                           delta_n_is, "IS-VaR-$(alpha_str)", log_scale)
            plot_estimates(n_is_samples, estimate_fn(:cvar, alpha, true),
                           delta_n_is, "IS-CVaR-$(alpha_str)", log_scale)
        end
    end
    xlabel("no. samples"); ylabel("estimates"); legend();
    if !isempty(output_dir)
        PyPlot.plt.savefig(joinpath(output_dir, "estimates.png"))
    end
end

# Plot histograms.
if plot_hist
    PyPlot.plt.figure(figsize=(9.0, 6.0))
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
    if !isempty(output_dir)
        PyPlot.plt.savefig(joinpath(output_dir, "histogram.png"))
    end
end
