# TODOs:
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
