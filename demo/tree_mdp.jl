# TODOs:
# - Move to src.
# - Try adding logprob in rollout.

struct TreeState
    name::String  # TOOD: Clean up.
    mdp_state::Any
end


# Initial state ctors.
TreeState(mdp_state::Any) = TreeState("TreeState", mdp_state)
TreeState(state::TreeState) = TreeState(state.mdp_state)


# Tree MDP type.
mutable struct TreeMDP <: MDP{TreeState, Any}
    rmdp::Any
    discount_factor::Float64
    distribution::Any
    reduction::String
end


function POMDPs.initialstate(mdp::TreeMDP)
    return TreeState(rand(initialstate(mdp.rmdp)))
end


function POMDPs.gen(m::TreeMDP, s::TreeState, action, rng)
    m_sp, cost = @gen(:sp, :r)(m.rmdp, s.mdp_state, action, rng)
    sp = TreeState(m_sp)
    return (sp=sp, r=cost)
end


function POMDPs.isterminal(mdp::TreeMDP, s::TreeState)
    return isterminal(mdp.rmdp, s.mdp_state)
end


POMDPs.discount(mdp::TreeMDP) = mdp.discount_factor


function POMDPs.actions(mdp::TreeMDP, s::TreeState)
    return mdp.distribution(mdp.rmdp, s.mdp_state)
end


function rollout(mdp::TreeMDP, s::TreeState, d::Int64,
                 cost::Float64, weight::Float64)
    if d == 0 || isterminal(mdp, s)
        return cost, weight
    else
        p_action = POMDPs.actions(mdp, s)
        action = rand(p_action)

        (sp, r) = @gen(:sp, :r)(mdp, s, action, Random.GLOBAL_RNG)
        new_cost = update_cost(cost, r, mdp.reduction)
        return rollout(mdp, sp, d - 1, cost, weight)
    end
end


function update_cost(acc_cost::Float64, new_cost::Union{Int64, Float64}, reduction::String)
    if reduction == "sum"
        return acc_cost + new_cost
    elseif reduction == "max"
        return max(acc_cost, new_cost)
    end
    throw("Not implemented reduction $(reduction)!")
end
