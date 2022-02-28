POMDPs.solve(solver::PISSolver, mdp::Union{POMDP,MDP}) = PISPlanner(solver, mdp)


"""
Deletes existing decision tree.
"""
function clear_tree!(p::PISPlanner)
    p.tree = nothing
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


"""
Calculates next action.
"""
function select_action(nodes, values, prob_α, cost_α, prob_p, n, α, β, γ)
    prob = adaptive_probs(values, prob_α, cost_α, prob_p, n, α, β, γ)
    sanode_idx = sample(1:length(nodes), Weights(prob))
    sanode = nodes[sanode_idx]
    q_logprob = log(prob[sanode_idx])
    return sanode, q_logprob
end


function adaptive_probs(values, prob_α, cost_α, prob_p, n, α, β, γ)
    cvar_strategy = [(cost_α[i] * prob_p[i]) for i=1:length(values)] .+ (max(cost_α...)) / 20 .+ 1e-5
    cdf_strategy = [(prob_α[i] * prob_p[i]) for i=1:length(values)] .+ (max(prob_α...)) * max(prob_p...) / 20 .+ 1e-5

    # Normalize to unity
    cvar_strategy /= sum(cvar_strategy)
    cdf_strategy /= sum(cdf_strategy)

    # Mixture weighting
    prob = β * prob_p .+ γ * cdf_strategy .+ (1 - β - γ) * cvar_strategy
    return prob
end


"""
Calculates importance sampling weights.
"""
function compute_weight(q_logprob, a, distribution)
    if distribution == nothing
        w = -q_logprob
    else
        w = logpdf(distribution, a) - q_logprob
    end
    return w
end


"""
Constructs a PISTree and choose an action.
"""
POMDPs.action(p::PISPlanner, s) = first(action_info(p, s))

function estimate_value(mdp::Union{POMDP,MDP}, state::TreeState,
                        depth::Int, cost::Float64, weight::Float64)
    return rollout(mdp, state, depth, cost, weight)
end

function next_action(gen::UniformActionGenerator, mdp::Union{POMDP,MDP}, s, snode::PISStateNode)
    rand(gen.rng, actions(mdp, s))
end

init_Q(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_Q(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Float64, n)

init_N(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_N(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Int, n)


"""
Constructs a PISTree and choose the best action.
"""
function POMDPModelTools.action_info(p::PISPlanner, s; tree_in_info=false, β=0.0, γ=1.0, schedule=0.1)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("MCTS cannot handle terminal states: s = $s")
        end

        tree = p.tree
        if !p.solver.keep_tree || tree == nothing
            tree = PISTree{statetype(p.mdp),actiontype(p.mdp)}()
            p.tree = tree
        end
        snode = insert_state_node!(tree, s)

        timer = p.solver.timer
        start_s = timer()
        timeout_s = start_s + p.solver.max_time
        n_iterations = p.solver.n_iterations
        p.solver.show_progress ? progress = Progress(n_iterations) : nothing

        sim_channel = Channel{Task}(min(1000, n_iterations)) do channel
            for n in 1:n_iterations
                put!(channel, Threads.@spawn simulate(p, snode, p.solver.depth, 0.0, 0.0,
                                                      β, γ, schedule, timeout_s))
            end
        end

        nquery = 0
        for sim_task in sim_channel
            if timer() > timeout_s
                p.solver.show_progress ? finish!(progress) : nothing
                @show Timeout reached
                break
            end

            try
                fetch(sim_task)  # Throws a TaskFailedException if failed.
                nquery += 1
                p.solver.show_progress ? next!(progress) : nothing
            catch err
                throw(err.task.exception)  # Throw the underlying exception.
            end
        end

        p.reset_callback(p.mdp, s)  # Optional: Leave the MDP in the current state.
        info[:search_time_s] = (timer() - start_s)
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode = best_sanode(tree, snode)
        a = sanode.a_label
    catch ex
        a = next_action(p.next_action, p.mdp, s, snode)
        info[:exception] = ex
    end

    return a, info
end


"""
Adds a sample to the tree.
"""
function add_sample!(tree::PISTree, cost::Float64, weight::Float64)
    push!(tree.costs, cost)
    push!(tree.weights, weight)
end


"""
Returns the reward for one iteration of MCTS.
"""
function simulate(dpw::PISPlanner, snode::PISStateNode, d::Int,
                  cost::Float64=0.0, weight::Float64=0.0,
                  β::Float64=0.0, γ::Float64=1.0, schedule::Float64=0.1,
                  timeout_s::Float64=0.0)
    sol = dpw.solver
    timer = sol.timer
    tree = dpw.tree
    s = snode.s_label

    Base.@lock tree.costs_weights_lock begin
        n_samples = length(tree.costs)
        for i in tree.cdf_est.last_i+1:n_samples
            # Depending on how fast/slow this is, making copies of costs and weights first then
            # immediately releasing the lock can be better. But, of course, as we collect more
            # samples, copying can increasingly be slow.
            ImportanceWeightedRiskMetrics.update!(tree.cdf_est, tree.costs[i], exp(tree.weights[i]))
        end
    end

    if isterminal(dpw.mdp, s)
        Base.@lock tree.costs_weights_lock begin
            add_sample!(tree, cost, weight)
        end
        return cost, weight
    elseif d == 0
        out_cost, out_weight = estimate_value(dpw.mdp, s, d, cost, weight)
        Base.@lock tree.costs_weights_lock begin
            add_sample!(tree, out_cost, out_weight)
        end
        return out_cost, out_weight
    end

    # Action progressive widening.
    if sol.enable_action_pw
        if n_children(snode) <= sol.k_action * total_n(snode)^sol.alpha_action
            a = next_action(dpw.next_action, dpw.mdp, s, snode)
            insert_action_node!(tree, snode, a,
                                init_N(sol.init_N, dpw.mdp, s, a),
                                init_Q(sol.init_Q, dpw.mdp, s, a))
        end
    elseif n_children(snode) == 0
        for a in support(actions(dpw.mdp, s))
            insert_action_node!(tree, snode, a,
                                init_N(sol.init_N, dpw.mdp, s, a),
                                init_Q(sol.init_Q, dpw.mdp, s, a))
        end
    end

    sanode, q_logprob = sample_sanode_UCB(dpw, snode, β, γ, schedule)
    a = sanode.a_label
    w_node = compute_weight(q_logprob, a, actions(dpw.mdp, s))

    # State progressive widening.
    spnode = nothing
    new_node = false
    if (n_a_children(sanode) <= sol.k_state * n(sanode)^sol.alpha_state) || n_a_children(sanode) == 0
        sp, r = @gen(:sp, :r)(dpw.mdp, s, a, dpw.rng)
        Base.@lock tree.state_nodes_lock begin
            if haskey(tree.state_nodes, sp) 
                spnode = tree.state_nodes[sp]
            else
                spnode = insert_state_node!(tree, sp)
                new_node = true
            end
        end
        Base.@lock sanode.a_lock begin
            push!(sanode.transitions, (sp, r))
            if !(sp in sanode.unique_transitions)
                push!(sanode.unique_transitions, sp)
                sanode.n_a_children += 1
            end
        end
    else
        sp, r = rand(dpw.rng, sanode.transitions)
        spnode = Base.@lock tree.state_nodes_lock begin; tree.state_nodes[sp]; end
    end

    new_weight = weight + w_node
    new_cost = update_cost(cost, r, dpw.mdp.reduction)
    if new_node
        out_cost, out_weight = estimate_value(dpw.mdp, sp, d - 1, new_cost, new_weight)
        Base.@lock tree.costs_weights_lock begin
            add_sample!(tree, out_cost, out_weight)
        end
        q = discount(dpw.mdp) * out_cost
    else
        out_cost, out_weight = simulate(
            dpw, spnode, d - 1, new_cost, new_weight, β, γ, schedule, timeout_s)
        q = discount(dpw.mdp) * out_cost
    end

    # Backpropgate and update node values.
    Base.@lock sanode.a_lock begin
        Base.@lock snode.s_lock begin
            snode.total_n += 1
            delete!(snode.a_selected, sanode.a_label)
            sanode.n += 1
            sanode.q += (q - sanode.q) / sanode.n
            ImportanceWeightedRiskMetrics.update!(
                sanode.conditional_cdf_est, q, exp(out_weight))
        end
    end

    return out_cost, out_weight
end


function sample_sanode_UCB(dpw::PISPlanner, snode::PISStateNode,
                           β::Float64, γ::Float64, schedule::Float64)
    mdp = dpw.mdp
    tree = dpw.tree
    sol = dpw.solver
    c = sol.exploration_constant
    virtual_loss = sol.virtual_loss
    α = sol.α

    all_UCB = []
    all_sanodes = []
    ltn = log(total_n(snode))
    cost_α = []
    prob_α = []
    prob_p = []
    all_α = []

    # Instead of creating a copy, we can presumably just compute the no. of children and read only
    # that many elements of children(snode). But, this is also technically not safe as the vector
    # can grow in size and be resized around the same time when we access the element.
    sa_children = Base.@lock snode.s_lock begin; deepcopy(children(snode)) end
    for a_label in sa_children
        Base.@lock tree.state_action_nodes_lock begin
            state_action_key = (snode.s_label, a_label)
            if !haskey(tree.state_action_nodes, state_action_key)
                continue
            end
            sanode = tree.state_action_nodes[state_action_key]
        end
        a_n = n(sanode)
        a_q = q(sanode)
        if (ltn <= 0 && a_n == 0) || c == 0.0
            UCB = a_q
        else
            UCB = a_q + c * sqrt(ltn / a_n)
        end

        vloss = Base.@lock snode.s_lock begin; (a_label in snode.a_selected ? virtual_loss : 0.0) end;
        UCB -= vloss

        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        
        push!(all_UCB, UCB)
        push!(all_sanodes, sanode)

        w_annealed = 1.0 / (1.0 + schedule * a_n)
        a_α = w_annealed + (1 - w_annealed) * α
        est_quantile = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, α)
        c_tail = ImportanceWeightedRiskMetrics.tail_cost(sanode.conditional_cdf_est, est_quantile)
        c_cdf = ImportanceWeightedRiskMetrics.cdf(sanode.conditional_cdf_est, est_quantile)
        push!(cost_α, c_tail)
        push!(prob_α, 1.0 - c_cdf)
        push!(prob_p, pdf(actions(mdp, snode.s_label), a_label))
        push!(all_α, a_α)
    end

    sanode, q_logprob = select_action(all_sanodes, all_UCB, prob_α, cost_α, prob_p, tree.cdf_est.last_i, all_α, β, γ)
    Base.@lock snode.s_lock begin; push!(snode.a_selected, sanode.a_label); end
    return sanode, q_logprob
end


"""
Returns the best action.
Some publications say to choose action that has been visited the most
e.g., Continuous Upper Confidence Trees by Couëtoux et al.
"""
function best_sanode(tree::PISTree, snode::PISStateNode)
    best_Q = -Inf
    best_sa = nothing
    for a_label in children(snode)
        state_action_key = (snode.s_label, a_label)
        sanode = tree.state_action_nodes[state_action_key]
        if q(sanode) > best_Q
            best_Q = q(sanode)
            best_sa = sanode
        end
    end
    return best_sa
end
