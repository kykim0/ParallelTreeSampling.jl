using FileIO
using Printf
using PyPlot

using ParallelTreeSampling

include("utils.jl")


# Parameters to be set for plotting.
mc_base_dir = "data/save"; is_base_dir = "data/save"
mc_data = [joinpath(mc_base_dir, "gwl_mc_100000_$(trial).jld2") for trial in 1:3]
# mc_data = []
is_data = [joinpath(is_base_dir, "gwl_is_p99_100000_$(trial).jld2") for trial in 1:3]
# is_data = []
# is_data2 = [joinpath(is_base_dir, "gwl_is_var_sigmoid_1353_100000_$(trial).jld2") for trial in 1:3]
is_data2 = []
max_num_mc = 100_000; max_num_is = -1; max_num_is2 = -1
delta_n_mc = 100; delta_n_is = 100; delta_n_is2 = 100
plot_est = true; plot_hist = false; plot_ws = false; log_scale = true; plot_tex = true
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


# Configure relevant params if for tex.
if plot_tex
    PyPlot.matplotlib[:rc]("font", family="serif")
    PyPlot.matplotlib[:rc]("text", usetex=true)
    PyPlot.matplotlib[:rc]("pgf", rcfonts=false)
end


# Plot convergence graphs.
alphas = [1e-2]
if plot_est
    PyPlot.plt.figure(figsize=(9.0, 6.0))
    for alpha in alphas
        alpha_str = @sprintf("%.1e", alpha)

        mc_var_ret = plot_estimates(n_mc_samples, estimate_fn(:var, alpha),
                                    delta_n_mc, "MC-VaR-$(alpha_str)", log_scale)
        is_var_ret = plot_estimates(n_is_samples, estimate_fn(:var, alpha),
                                    delta_n_is, "IS-VaR-$(alpha_str)", log_scale)
        is2_var_ret = plot_estimates(n_is2_samples, estimate_fn(:var, alpha),
                                     delta_n_is, "IS2-VaR-$(alpha_str)", log_scale)

        mc_cvar_ret = plot_estimates(n_mc_samples, estimate_fn(:cvar, alpha),
                                     delta_n_mc, "MC-CVaR-$(alpha_str)", log_scale)
        is_cvar_ret = plot_estimates(n_is_samples, estimate_fn(:cvar, alpha),
                                     delta_n_is, "IS-CVaR-$(alpha_str)", log_scale)
        is2_cvar_ret = plot_estimates(n_is2_samples, estimate_fn(:cvar, alpha),
                                      delta_n_is, "IS2-CVaR-$(alpha_str)", log_scale)
    end
    xlabel("Samples"); ylabel("Estimates"); legend();
    if !isempty(output_dir)
        PyPlot.plt.tight_layout()
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

    ylim(bottom=0.0); xlabel("Costs"); ylabel("Density"); legend();
    if !isempty(output_dir)
        PyPlot.plt.tight_layout()
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
        PyPlot.plt.tight_layout()
        PyPlot.plt.savefig(joinpath(output_dir, "sample-weights.png"), dpi=500)
    end
end
