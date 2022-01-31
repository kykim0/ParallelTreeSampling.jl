# Based on TreeImportanceSampling.jl.

struct TreeState
    values::Vector{Any}  # Multi-level state.
    costs::Vector{Any}
    mdp_state::Any
    done::Bool
    w::Float64  # Importance sampling weight.
end


# Initial state ctors.
TreeState(mdp_state::Any) = TreeState([], [0.0], mdp_state, false, 0.0)
TreeState(state::TreeState, w::Float64) = TreeState(state.values, state.costs, state.mdp_state, state.done, w)


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
    if !state.done
        r = 0
    else
        if mdp.reduction=="sum"
            r = sum(state.costs)
        elseif mdp.reduction=="max"
            r = max(state.costs...)
        else
            throw("Not implemented reduction $(mdp.reduction)!")
        end
        push!(mdp.costs, r)
        push!(mdp.IS_weights, state.w)
    end
    return r
end


function POMDPs.initialstate(mdp::TreeMDP)  # rng unused.
    return TreeState(rand(initialstate(mdp.rmdp)))
end


function POMDPs.gen(m::TreeMDP, s::TreeState, a, rng)
    # transition model
    m_sp, cost = @gen(:sp, :r)(m.rmdp, s.mdp_state, first(a), rng)
    sp = TreeState([s.values..., first(a)], [s.costs..., cost], m_sp, isterminal(m.rmdp, m_sp), last(a))

    r = POMDPs.reward(m, sp, a)
    return (sp=sp, r=r)
end


function POMDPs.isterminal(mdp::TreeMDP, s::TreeState)
    return s.done
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
        q_value = r + discount(mdp)*first(rollout(mdp, sp, w, d-1))

        return q_value, w
    end
end
