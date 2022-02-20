# TODOs:
# - Move to src.

struct TreeState
    values::Vector{Any}  # Multi-level state.
    costs::Vector{Any}
    mdp_state::Any
    w::Float64  # Importance sampling weight.
end


# Initial state ctors.
TreeState(mdp_state::Any) = TreeState([], [0.0], mdp_state, 0.0)
TreeState(state::TreeState, w::Float64) = TreeState(state.values, state.costs, state.mdp_state, w)


# Tree MDP type.
mutable struct TreeMDP <: MDP{TreeState, Any}
    rmdp::Any
    discount_factor::Float64
    costs::Vector
    IS_weights::Vector
    distribution::Any
    reduction::String
end


function POMDPs.reward(mdp::TreeMDP, state::TreeState, action)
    if !isterminal(mdp, state)
        r = 0
    else
        if mdp.reduction == "sum"
            r = sum(state.costs)
        elseif mdp.reduction == "max"
            r = maximum(state.costs)
        else
            throw("Not implemented reduction $(mdp.reduction)!")
        end
        push!(mdp.costs, r)
        push!(mdp.IS_weights, state.w)
    end
    return r
end


function POMDPs.initialstate(mdp::TreeMDP)
    return TreeState(rand(initialstate(mdp.rmdp)))
end


function POMDPs.gen(m::TreeMDP, s::TreeState, action, rng)
    a, w = action
    m_sp, cost = @gen(:sp, :r)(m.rmdp, s.mdp_state, a, rng)
    sp = TreeState([s.values..., a], [s.costs..., cost], m_sp, w)

    r = POMDPs.reward(m, sp, action)
    return (sp=sp, r=r)
end


function POMDPs.isterminal(mdp::TreeMDP, s::TreeState)
    return isterminal(mdp.rmdp, s.mdp_state)
end


POMDPs.discount(mdp::TreeMDP) = mdp.discount_factor


function POMDPs.actions(mdp::TreeMDP, s::TreeState)
    return mdp.distribution(mdp.rmdp, s.mdp_state)
end


function rollout(mdp::TreeMDP, s::TreeState, w::Float64, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0, w
    else
        p_action = POMDPs.actions(mdp, s)
        a = rand(p_action)

        (sp, r) = @gen(:sp, :r)(mdp, s, [a, w], Random.GLOBAL_RNG)
        q_value = r + discount(mdp) * first(rollout(mdp, sp, w, d-1))

        return q_value, w
    end
end
