# Updates the accumulated cost.
function update_cost(acc_cost::Float64, new_cost::Union{Int64,Float64},
                     reduction_type::Symbol)
    if reduction_type == :sum
        return acc_cost + new_cost
    elseif reduction_type == :max
        return max(acc_cost, new_cost)
    end
    throw("Unsupported cost reduction strategy: $(reduction_type)")
end


# Numerically stable softmax.
function softmax(x, θ::AbstractFloat=1.0, dim::Integer=1)
    x_exps = exp.((x .- maximum(x)) * θ)
    return x_exps ./ sum(x_exps, dims=dim)
end


# Uniform probability vector of a given length.
uniform_probs(n::Integer) = fill(1 / n, n)

# Various strategies for computing action selection probabilities.
# TODO(kykim): Consider putting these out into a separate jl.
function ucb_probs(ucb_scores)
    if all(isapprox(e, 0.0) for e in ucb_scores)
        return uniform_probs(length(ucb_scores))
    end
    scores = ucb_scores .- minimum(ucb_scores)
    scores /= sum(scores)
    return scores
end


function ucb_softmax_probs(ucb_scores)
    return softmax(ucb_scores, 1.0)
end


function expected_cost_probs(est_α_probs, est_α_costs)
    if all(isapprox(c, 0.0) for c in est_α_costs)
        return uniform_probs(length(est_α_costs))
    end
    scores = est_α_probs .* est_α_costs
    scores /= sum(scores)
    return scores
end


function var_sigmoid(est_var, est_α_probs, est_α_costs, nominal_probs)
    if all(isapprox(c, 0.0) for c in est_α_costs)
        return uniform_probs(length(est_α_costs))
    end

    # TODO(kykim): Divide by the current worst cost?
    norm_costs = (est_α_costs .- est_var) / (maximum(est_α_costs) + eps())
    @assert !any(isnan(c) for c in norm_costs) "norm_costs NaN $(norm_costs)"
    scores = 1 ./ (1 .+ exp.(-norm_costs))  # Sigmoid centered at est_var.
    @assert !any(isnan(s) for s in scores) "scores NaN $(scores)"
    return scores ./ sum(scores)
end


function mixture_probs(est_α_probs, est_α_costs, nominal_probs, γ)
    var_distrib = nominal_probs .* est_α_probs .+ eps()
    cvar_distrib = nominal_probs .* est_α_probs .* est_α_costs .+ eps()

    var_distrib /= sum(var_distrib)
    cvar_distrib /= sum(cvar_distrib)

    # Mixture weighting.
    mixture_distrib = γ * var_distrib .+ (1 - γ) * cvar_distrib
    mixture_distrib /= sum(mixture_distrib)
    return mixture_distrib
end


function adaptive_probs(est_α_probs, est_α_costs, nominal_probs, β, γ)
    max_α_cost = maximum(est_α_costs)
    max_α_prob = maximum(est_α_probs)
    max_nominal_prob = maximum(nominal_probs)

    cvar_strategy = est_α_costs .* nominal_probs .+ (max_α_cost / 20) .+ eps()
    cdf_strategy = est_α_probs .* nominal_probs .+ (max_α_prob * max_nominal_prob / 20) .+ eps()
    cvar_strategy /= sum(cvar_strategy)
    cdf_strategy /= sum(cdf_strategy)

    # Mixture weighting.
    a_probs = β * nominal_probs .+ γ * cdf_strategy .+ (1 - β - γ) * cvar_strategy
    return a_probs
end
