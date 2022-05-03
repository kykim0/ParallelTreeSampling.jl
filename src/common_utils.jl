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


# Various strategies for computing action selection probabilities.
# TODO(kykim): Consider putting these out into a separate jl.
function ucb_probs(ucb_scores)
    # In case all zeros, transform the vector to do uniform sampling.
    if all(isapprox(e, 0.0) for e in ucb_scores)
        scores = fill(1 / length(ucb_scores), length(ucb_scores))
    else
        scores = ucb_scores .- minimum(ucb_scores)
    end
    return scores /= sum(scores)
end


function ucb_softmax_probs(ucb_scores)
    return softmax(ucb_scores, 1.0)
end


function expected_cost_probs(est_α_probs, est_α_costs)
    exp_α_costs = est_α_probs .* est_α_costs
    return exp_α_costs /= sum(exp_α_costs)
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
