"""
Updates the accumulated cost.
"""
function update_cost(acc_cost::Float64, new_cost::Union{Int64, Float64}, reduction::String)
    if reduction == "sum"
        return acc_cost + new_cost
    elseif reduction == "max"
        return max(acc_cost, new_cost)
    end
    throw("Not implemented reduction $(reduction)!")
end


"""
Utility function for numerically stable softmax.
"""
_exp(x) = exp.(x .- maximum(x))
_exp(x, θ::AbstractFloat) = exp.((x .- maximum(x)) * θ)
_sftmax(e, d::Integer) = (e ./ sum(e, dims = d))

function softmax(X, dim::Integer)
    _sftmax(_exp(X), dim)
end

function softmax(X, dim::Integer, θ::Float64)
    _sftmax(_exp(X, θ), dim)
end

softmax(X) = softmax(X, 1)
