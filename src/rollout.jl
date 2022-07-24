function rollout(mdp::Union{POMDP,MDP}, s, d::Int64,
                 cost::Float64, weight::Float64,
                 action_distrib_fn::Function, strategy::Symbol)
    if d == 0 || isterminal(mdp, s)
        return cost, weight
    else
        action, action_w = sim_action(mdp, s, action_distrib_fn, strategy)
        weight += action_w

        (sp, r) = @gen(:sp, :r)(mdp, s, action, Random.GLOBAL_RNG)
        new_cost = update_cost(cost, r, :sum)
        return rollout(mdp, sp, d - 1, new_cost, weight,
                       action_distrib_fn, strategy)
    end
end


function estimate_value(mdp::Union{POMDP,MDP}, state, depth::Int,
                        cost::Float64, weight::Float64,
                        action_distrib_fn::Function, strategy::Symbol)
    return rollout(mdp, state, depth, cost, weight,
                   action_distrib_fn, strategy)
end


# Chooses an action for during a rollout depending on the strategy.
function sim_action(mdp::Union{POMDP,MDP}, s, action_distrib_fn, strategy::Symbol)
    p_action = action_distrib_fn(mdp, s)
    if strategy == :uniform
        n_action = length(support(p_action)); rand_idx = rand(1:n_action)
        action = support(p_action)[rand_idx]
        action_w = logpdf(p_action, action) - log(1 / n_action)
        return action, action_w
    elseif strategy == :nominal
        action = rand(p_action)
        # Note that the weight is not updated as we are sampling from the
        # nominal, and it is the logprob i.e., log(1.0) = 0.0.
        return action, 0.0
    end
    throw("Unsupported rollout strategy: $(strategy)")
end


function next_action(gen::UniformActionGenerator, mdp::Union{POMDP,MDP}, s)
    rand(gen.rng, actions(mdp, s))
end


init_Q(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_Q(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Float64, n)


init_N(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_N(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Int, n)
