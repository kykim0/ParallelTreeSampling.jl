# Updates the accumulated cost.
function update_cost(acc_cost::Float64, new_cost::Union{Int64,Float64},
                     reduction_type::Symbol=:sum)
    if reduction_type == :sum
        return acc_cost + new_cost
    elseif reduction_type == :max
        return max(acc_cost, new_cost)
    end
    throw("Unsupported cost reduction strategy: $(reduction_type)")
end


# Returns a linear decay schedule function.
function linear_decay_schedule(start_w::Float64, end_w::Float64, end_n)
    fn = function(curr_n)
        curr_n >= end_n && return end_w
        slope = (end_w - start_w) / (end_n - 1)
        return start_w + slope * (curr_n - 1)
    end
    return fn
end


# Returns an exp decay schedule function.
function exp_decay_schedule(start_w::Float64, end_w::Float64, end_n)
    fn = function(curr_n)
        curr_n >= end_n && return end_w
        ratio = exp(log(end_w / start_w) / (end_n - 1))
        return start_w * ratio^(curr_n - 1)
    end
    return fn
end
