# Various strategies for computing action selection probabilities.

# Numerically stable softmax.
function softmax(x, θ::AbstractFloat=1.0, dim::Integer=1)
    x_exps = exp.((x .- maximum(x)) * θ)
    return x_exps ./ sum(x_exps, dims=dim)
end

# Uniform probability vector of a given length.
uniform_probs(n::Integer) = fill(1 / n, n)


function ucb_probs(ucb_scores, nominal_probs)
    if all(isapprox(e, 0.0) for e in ucb_scores)
        return nominal_probs
    end
    scores = ucb_scores .- minimum(ucb_scores)
    scores /= sum(scores)
    return scores
end


function ucb_softmax_probs(ucb_scores)
    return softmax(ucb_scores, 1.0)
end


function var_sigmoid(est_var, est_worst, ucb_scores, nominal_probs,
                     emp_var, mix_w, min_s)
    if any(isapprox(c, 0.0) for c in ucb_scores)
        return uniform_probs(length(ucb_scores))
    end

    norm_costs = ucb_scores .- est_var
    # slope = est_worst + eps()
    # slope = maximum(ucb_scores) - est_var + eps()
    p = 0.95
    slope = max(-sqrt(emp_var) / log((1 - p) / p), min_s)
    sigmoid_scores = 1 ./ (1 .+ exp.(-norm_costs / slope))  # Sigmoid centered at est_var.

    var_scores = sigmoid_scores .* nominal_probs
    var_scores /= sum(var_scores)
    cvar_scores = sigmoid_scores .* nominal_probs .* max.(norm_costs, eps())
    cvar_scores /= sum(cvar_scores)
    var_w = mix_w; cvar_w = 1.0 - var_w
    scores = var_w * var_scores + cvar_w * cvar_scores
    return scores ./ sum(scores)
end


function mixture_probs(est_α_probs, est_α_costs, nominal_probs, γ=0.3)
    var_distrib = nominal_probs .* est_α_probs .+ eps()
    cvar_distrib = nominal_probs .* est_α_probs .* est_α_costs .+ eps()

    var_distrib /= sum(var_distrib)
    cvar_distrib /= sum(cvar_distrib)

    # Mixture weighting.
    mixture_distrib = γ * var_distrib .+ (1 - γ) * cvar_distrib
    mixture_distrib /= sum(mixture_distrib)
    return mixture_distrib
end


function adaptive_probs(est_α_probs, est_α_costs, nominal_probs, β=0.1, γ=0.3)
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
