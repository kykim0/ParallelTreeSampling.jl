"""
Deletes existing decision tree.
"""
function clear_tree!(p::PISPlanner)
    p.tree = nothing
end


"""
Constructs a PISTree and choose an action.
"""
POMDPs.action(p::PISPlanner, s) = first(action_info(p, s))

"""
Constructs a PISTree and choose the best action.
"""
function POMDPModelTools.action_info(p::PISPlanner, s; tree_in_info=false, kwargs...)
    local a::Any  # actiontype(p.mdp)
    info = Dict{Symbol, Any}()

    if isterminal(p.mdp, s)
        error("MCTS cannot handle terminal states: s = $s")
    end

    tree = p.tree
    if !p.solver.keep_tree || isnothing(tree)
        # TODO(kykim): Temporary hack to get around the RMDP case.
        # Should ideally be actiontype(p.mdp).
        tree = PISTree{statetype(p.mdp),Any}()
        p.tree = tree
    end
    snode = insert_state_node!(tree, s)

    timer = p.solver.timer
    start_s = timer()
    timeout_s = start_s + p.solver.max_time
    n_iterations = p.solver.n_iterations
    p.solver.show_progress ? progress = Progress(n_iterations) : nothing

    # TODO(kykim): Implement DM weighting.
    sim_channel = Channel{Task}(min(10_000, n_iterations)) do channel
        for n in 1:n_iterations
            put!(channel, Threads.@spawn simulate_sample(
                p, snode, n, timeout_s; kwargs...))
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
    if tree_in_info
        info[:tree] = tree
    end

    sanode = best_sanode(tree, snode)
    a = sanode.a_label

    return a, info
end


"""
Simulates one sample.
"""
function simulate_sample(dpw::PISPlanner, snode::PISStateNode,
                         iter_n::Integer, timeout_s::Float64=0.0; kwargs...)
    tree = dpw.tree

    kwargs = Dict(kwargs)
    α = get(kwargs, :α, 0.1)
    β = get(kwargs, :β, 0.0)
    γ = get(kwargs, :γ, 1.0)
    schedule = get(kwargs, :schedule, 0.0)

    d = dpw.solver.depth
    est_var = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, α)
    est_worst = tree.cdf_est.Xs[end]
    # TODO(kykim): Add a struct to group all these args to simulate().
    cost, weight = simulate(dpw, snode, d, 0.0, 0.0, iter_n, timeout_s;
                            est_var, est_worst, α, β, γ, schedule)

    Base.@lock tree.costs_weights_lock begin
        push!(tree.costs, cost)
        push!(tree.weights, weight)

        # TODO(kykim): Try updating the tree cdf only once every n iterations.
        n_samples = length(tree.costs)
        for i in tree.cdf_est.last_i+1:n_samples
            # Depending on how fast/slow this is, making copies of costs and weights first then
            # immediately releasing the lock can be better. But, of course, as we collect more
            # samples, copying can increasingly be slow.
            ImportanceWeightedRiskMetrics.update!(tree.cdf_est, tree.costs[i], exp(tree.weights[i]))
        end
    end
end


"""
Returns the reward for one iteration of MCTS.
"""
function simulate(dpw::PISPlanner, snode::PISStateNode, d::Int,
                  cost::Float64=0.0, weight::Float64=0.0,
                  iter_n::Integer=0, timeout_s::Float64=0.0;
                  est_var::Float64, est_worst::Float64, α::Float64, β::Float64,
                  γ::Float64, schedule::Float64)
    tree = dpw.tree
    sol = dpw.solver
    s = snode.s_label
    exp_config = sol.experiment_config
    action_distrib_fn = sol.nominal_distrib_fn
    action_distrib = action_distrib_fn(dpw.mdp, s)

    if isterminal(dpw.mdp, s)
        return cost, weight
    elseif d == 0
        out_cost, out_weight = estimate_value(dpw.mdp, s, d, cost, weight,
                                              sol.cost_reduction,
                                              action_distrib_fn)
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

    # Compute various action statistics.
    sanodes, ucb_scores, est_α_probs, est_α_costs, nominal_probs = sanode_stats(
        dpw, snode, action_distrib, est_var, α, schedule)

    # Sample an action based on the stats and action selection method.
    nominal_steps = exp_config.nominal_steps
    a_select = iter_n < nominal_steps ? :nominal : sol.action_selection
    action_probs = compute_action_probs(a_select, est_var, est_worst, ucb_scores,
                                        est_α_probs, est_α_costs, nominal_probs,
                                        β, γ)
    sanode, q_logprob = select_action(sanodes, action_probs)
    Base.@lock snode.s_lock begin; push!(snode.a_selected, sanode.a_label); end

    a = sanode.a_label
    w_node = compute_weight(q_logprob, a, action_distrib)

    # State progressive widening.
    spnode = nothing
    new_node = false
    if ((sol.enable_state_pw && n_a_children(sanode) <= sol.k_state * n(sanode)^sol.alpha_state) ||
        n_a_children(sanode) == 0)
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
    new_cost = update_cost(cost, r, sol.cost_reduction)
    if new_node
        out_cost, out_weight = estimate_value(dpw.mdp, sp, d - 1, new_cost,
                                              new_weight, sol.cost_reduction,
                                              action_distrib_fn)
        q = discount(dpw.mdp) * out_cost
    else
        out_cost, out_weight = simulate(
            dpw, spnode, d - 1, new_cost, new_weight, iter_n, timeout_s;
            est_var, est_worst, α, β, γ, schedule)
        q = discount(dpw.mdp) * out_cost
    end

    # Backpropagate and update node values.
    Base.@lock sanode.a_lock begin
        Base.@lock snode.s_lock begin
            snode.total_n += 1
            delete!(snode.a_selected, sanode.a_label)
            sanode.n += 1
            sanode.q += (q - sanode.q) / sanode.n
            ImportanceWeightedRiskMetrics.update!(sanode.c_cdf_est, q, 1.0)
        end
    end

    return out_cost, out_weight
end


"""
Computes importance weights i.e., p/q.
"""
function compute_weight(q_logprob, a, distribution)
    if isnothing(distribution)
        w = -q_logprob
    else
        w = logpdf(distribution, a) - q_logprob
    end
    return w
end


"""
Returns a sampled action and the corresponding log probability.
"""
function select_action(sanodes, action_probs)
    sanode_idx = sample(1:length(sanodes), Weights(Float64.(action_probs)))
    sanode = sanodes[sanode_idx]
    q_logprob = log(action_probs[sanode_idx])
    @assert !isnan(q_logprob) "q_logprob NaN $(action_probs)"
    return sanode, q_logprob
end


"""
Computes the probabilities with which to sample from the actions.
"""
function compute_action_probs(a_selection::Symbol, est_var, est_worst,
                              ucb_scores, est_α_probs, est_α_costs,
                              nominal_probs, β, γ)
    a_selection == :nominal && return nominal_probs
    a_selection == :ucb && return ucb_probs(ucb_scores)
    a_selection == :ucb_softmax && return ucb_softmax_probs(ucb_scores)
    a_selection == :expected_cost &&
        return expected_cost_probs(est_α_probs, est_α_costs)
    a_selection == :mixture &&
        return mixture_probs(est_α_probs, est_α_costs, nominal_probs, γ)
    a_selection == :var_sigmoid &&
        return var_sigmoid(est_var, est_worst, ucb_scores, nominal_probs)
    a_selection == :adaptive &&
        return adaptive_probs(est_α_probs, est_α_costs, nominal_probs, β, γ)
    # TODO(kykim): Boltzmann exploration type strategy.
    throw("Unsupported action selection strategy: $(a_selection)")
end


"""
Returns an adaptive α to use for an action.
"""
function action_α(α, a_n, schedule)
    schedule == 0.0 && return α
    w_annealed = 1.0 / (1.0 + schedule * a_n)
    return w_annealed + (1 - w_annealed) * α
end


function sanode_stats(dpw::PISPlanner, snode::PISStateNode, action_distrib,
                      est_var::Float64, α::Float64, schedule::Float64)
    tree = dpw.tree
    sol = dpw.solver
    c = sol.exploration_constant
    virtual_loss = sol.virtual_loss

    ucb_scores = []
    sanodes = []
    ltn = log(total_n(snode))
    est_α_costs = []
    est_α_probs = []
    nominal_probs = []

    # α schedule for potentially smoother convergence.
    a_α = action_α(α, 1.0, 0.0)

    # Instead of creating a copy, we can presumably just compute the no. of children and read only
    # that many elements of children(snode). But, this is also technically not safe as the vector
    # can grow in size and be resized around the same time when we access the element.
    sa_children = Base.@lock snode.s_lock begin; deepcopy(children(snode)) end
    for a_label in sa_children
        sanode = nothing
        Base.@lock tree.state_action_nodes_lock begin
            state_action_key = sanode_key(snode.s_label, a_label)
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

        # If applicable, apply virtual loss to the score.
        vloss = Base.@lock snode.s_lock begin; (a_label in snode.a_selected ? virtual_loss : 0.0) end;
        UCB -= vloss
        
        @assert !isnan(UCB) "UCB was NaN (q=$a_q, c=$c, ltn=$ltn, n=$a_n)"
        @assert !isequal(UCB, -Inf) "UCB was -Inf (q=$a_q, c=$c, ltn=$ltn, n=$a_n)"
        
        push!(ucb_scores, UCB)
        push!(sanodes, sanode)

        # TODO(kykim): Should the scheduling be really at the action level i.e.
        #   a_α = action_α(α, a_n, schedule)
        #   est_var = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, a_α)
        c_tail = ImportanceWeightedRiskMetrics.tail_cost(sanode.c_cdf_est, est_var)
        c_cdf = ImportanceWeightedRiskMetrics.cdf(sanode.c_cdf_est, est_var)
        push!(est_α_costs, c_tail)
        push!(est_α_probs, 1.0 - c_cdf)
        push!(nominal_probs, pdf(action_distrib, a_label))
    end

    return sanodes, ucb_scores, est_α_probs, est_α_costs, nominal_probs
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
        state_action_key = sanode_key(snode.s_label, a_label)
        sanode = tree.state_action_nodes[state_action_key]
        if q(sanode) > best_Q
            best_Q = q(sanode)
            best_sa = sanode
        end
    end
    return best_sa
end
