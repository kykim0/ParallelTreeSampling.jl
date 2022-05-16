using FileIO
using Printf
using PyPlot

using ParallelTreeSampling

include("utils.jl")


# Parameters to be set for plotting.
mc_base_dir = "data/2022-05-15"; is_base_dir = "data/2022-05-15"
# mc_data = [joinpath(mc_base_dir, "gwl_mc_1746_10000000_$(trial).jld2") for trial in 1:3]
mc_data = []
is_data = [joinpath(is_base_dir, "gwl_is_var_sigmoid_1e-3_1742_100000_$(trial).jld2") for trial in 1:3]
# is_data = []
is_data2 = [joinpath(is_base_dir, "gwl_is_adaptive_2036_100000_$(trial).jld2") for trial in 1:3]
# is_data2 = []
max_num_mc = 1_000_000; max_num_is = -1; max_num_is2 = -1
delta_n_mc = 100; delta_n_is = 100; delta_n_is2 = 100
plot_est = true; plot_hist = true; plot_ws = false; log_scale = true
output_dir = joinpath(homedir(), "Desktop")

# Constants to be updated as needed.
samples_key = "risks:"; weights_key = "weights:"

# Read the jld2 files.
function load_jld2(jld2_filename)
    d = load(jld2_filename)
    @assert haskey(d, samples_key)
    return (haskey(d, weights_key) ?
        (d[samples_key], exp.(d[weights_key])) : d[samples_key])
end
n_mc_samples = [load_jld2(mc_datum)[1:(max_num_mc < 0 ? end : max_num_mc)]
                for mc_datum in mc_data]
n_is_samples = [load_jld2(is_datum)[1:(max_num_is < 0 ? end : max_num_is)]
                for is_datum in is_data]
n_is2_samples = [load_jld2(is_datum)[1:(max_num_is < 0 ? end : max_num_is)]
                 for is_datum in is_data2]


# Returns a lambda for computing metrics used for plotting.
#
# Valid metric types include :mean, :var, :cvar, :worst.
function estimate_fn(metric_type, alpha)
    fn = function(samples, weights=nothing)
        metrics = eval_metrics(samples; weights=weights, alpha=alpha)
        return getproperty(metrics, metric_type)
    end
    return fn
end


# Plot convergence graphs.
alphas = [1e-3]
if plot_est
    PyPlot.plt.figure(figsize=(9.0, 6.0))
    for alpha in alphas
        alpha_str = @sprintf("%.1e", alpha)

        plot_estimates(n_mc_samples, estimate_fn(:var, alpha),
                       delta_n_mc, "MC-VaR-$(alpha_str)", log_scale)
        plot_estimates(n_mc_samples, estimate_fn(:cvar, alpha),
                       delta_n_mc, "MC-CVaR-$(alpha_str)", log_scale)

        plot_estimates(n_is_samples, estimate_fn(:var, alpha),
                       delta_n_is, "IS-VaR-$(alpha_str)", log_scale)
        plot_estimates(n_is_samples, estimate_fn(:cvar, alpha),
                       delta_n_is, "IS-CVaR-$(alpha_str)", log_scale)

        plot_estimates(n_is2_samples, estimate_fn(:var, alpha),
                       delta_n_is, "IS2-VaR-$(alpha_str)", log_scale)
        plot_estimates(n_is2_samples, estimate_fn(:cvar, alpha),
                       delta_n_is, "IS2-CVaR-$(alpha_str)", log_scale)
    end
    xlabel("no. samples"); ylabel("estimates"); legend();
    if !isempty(output_dir)
        PyPlot.plt.savefig(joinpath(output_dir, "estimates.png"), dpi=500)
    end
end

# Plot histograms.
if plot_hist
    PyPlot.plt.figure(figsize=(9.0, 6.0))

    nbins = 50
    samples = isempty(n_mc_samples) ? [] : first(n_mc_samples)
    plot_histogram(samples, bins=nbins, label="MC")

    samples, weights = isempty(n_is_samples) ? ([], []) : first(n_is_samples)
    plot_histogram(samples, bins=nbins, label="IS-uw")
    plot_histogram(samples, weights, bins=nbins, label="IS-w")

    samples, weights = isempty(n_is2_samples) ? ([], []) : first(n_is2_samples)
    plot_histogram(samples, bins=nbins, label="IS2-uw")
    plot_histogram(samples, weights, bins=nbins, label="IS2-w")

    ylim(bottom=0.0); xlabel("costs"); ylabel("density"); legend();
    if !isempty(output_dir)
        PyPlot.plt.savefig(joinpath(output_dir, "histogram.png"), dpi=500)
    end
end

# Plot samples of weights.
if plot_ws
    PyPlot.plt.figure(figsize=(9.0, 6.0))

    n = 100
    samples, weights = first(n_is_samples)
    plot_samples(weights, n=n, label="IS-weights")

    samples, weights = first(n_is2_samples)
    plot_samples(weights, n=n, label="IS2-weights")

    xlabel("i"); ylabel("weights"); legend();
    if !isempty(output_dir)
        PyPlot.plt.savefig(joinpath(output_dir, "sample-weights.png"), dpi=500)
    end
end
