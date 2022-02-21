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
