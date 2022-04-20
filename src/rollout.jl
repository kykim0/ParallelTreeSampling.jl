function rollout(mdp::Union{POMDP,MDP}, s, d::Int64,
                 cost::Float64, weight::Float64, w_reduction::String,
                 action_distrib_fn::Function)
    if d == 0 || isterminal(mdp, s)
        return cost, weight
    else
        p_action = action_distrib_fn(mdp, s)
        action = rand(p_action)

        (sp, r) = @gen(:sp, :r)(mdp, s, action, Random.GLOBAL_RNG)
        new_cost = update_cost(cost, r, w_reduction)
        return rollout(mdp, sp, d - 1, cost, weight, w_reduction,
                       action_distrib_fn)
    end
end


function estimate_value(mdp::Union{POMDP,MDP}, state, depth::Int,
                        cost::Float64, weight::Float64, w_reduction::String,
                        action_distrib_fn::Function)
    return rollout(mdp, state, depth, cost, weight, w_reduction,
                   action_distrib_fn)
end


function next_action(gen::UniformActionGenerator, mdp::Union{POMDP,MDP}, s, snode::PISStateNode)
    rand(gen.rng, actions(mdp, s))
end


init_Q(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_Q(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Float64, n)


init_N(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_N(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Int, n)
