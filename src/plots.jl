using PyPlot


# Plots incremental sampling estimates with a confidence region.
#
# Args:
#   n_samples: a Vector of independent sets of samples.
#   estimate_fn: a lambda that takes as input a Vector of samples (and
#     additionally a Vector of weights) and returns a scalar metric.
#   delta_n: compute incremental estimates every this many steps.
#   label: a String label to use for the plot.
#   log_scale: true to set x on the log scale.
function plot_estimates(n_samples::Vector, estimate_fn::Function,
                        delta_n::Integer, label::String, log_scale::Bool=false)
    if isempty(n_samples); return nothing; end

    # Compute the x range.
    if isa(first(n_samples), Tuple)
        total_n = length(first(n_samples)[1])
    else
        total_n = length(first(n_samples))
    end
    xl = []
    x_i = delta_n
    while x_i < total_n
        push!(xl, x_i)
        if log_scale
            if 3 * x_i < total_n; push!(xl, 3 * x_i); end
            x_i *= 10
        else
            x_i += delta_n
        end
    end
    if (last(xl) != total_n); push!(xl, total_n); end

    # Compute incremental estimates.
    n_y = [[isa(samples, Tuple) ?
        estimate_fn(samples[1][1:x], samples[2][1:x]) :
        estimate_fn(samples[1:x]) for samples in n_samples] for x in xl]
    min_y, max_y = minimum.(n_y), maximum.(n_y)
    mid_y = (max_y + min_y) ./ 2.0

    # For Plots.jl:
    #  plot(x_mc, mid_y_mc, ribbon=(max_y_mc - mid_y_mc), fillalpha=0.15,
    #       label="MC", lw=2, xlabel="no. of samples", ylabel="estimates")
    #  plot(x_is, mid_y_is, ribbon=(max_y_is - mid_y_is), fillalpha=0.15, label="IS", lw=2)
    PyPlot.plt.plot(xl, mid_y, label=label, lw=0.5)
    log_scale && PyPlot.plt.xscale("log")
    PyPlot.plt.fill_between(xl, min_y, max_y, alpha=0.15)
    return xl, min_y, mid_y, max_y
end


# Plots a histogram of samples drawn.
function plot_histogram(samples::Vector, weights::Union{Vector,Nothing}=nothing;
                        bins::Integer=10, density::Bool=true,
                        label::String=nothing)
    if isempty(samples); return; end

    PyPlot.plt.hist(samples, bins, weights=weights, density=density,
                    label=label, alpha=0.3)
end


# Plots bar graphs of a series of samples.
function plot_samples(samples::Vector; n::Integer, label::String=nothing)
    if isempty(samples); return; end

    # Create a subset of samples of size roughly n.
    delta_n = Int(floor(length(samples) / n))
    samples = samples[1:delta_n:end]
    xs = [x for x in 1:length(samples)]

    PyPlot.plt.bar(xs, samples, label=label)
end
