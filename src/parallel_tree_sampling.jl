POMDPs.solve(solver::PISDPWSolver, mdp::Union{POMDP,MDP}) = PISDPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::PISDPWPlanner)
    p.tree = nothing
end

"""
Utility function for numerically stable softmax 
Adapted from: https://nextjournal.com/jbowles/the-mathematical-ideal-and-softmax-in-julia
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
Calculate next action
"""
function select_action(nodes, values; c=5.0)
    prob = softmax(c*values)
    sanode_idx = sample(1:length(nodes), Weights(prob))
    sanode = nodes[sanode_idx]
    q_logprob = log(prob[sanode_idx])
    return sanode, q_logprob
end

"""
Calculate IS weights
"""
function compute_IS_weight(q_logprob, a, distribution)
    if distribution == nothing
        w = -q_logprob
    else
        w = logpdf(distribution, a) - q_logprob
    end
    return w
end

"""
Construct an PISDPW tree and choose an action.
"""
POMDPs.action(p::PISDPWPlanner, s) = first(action_info(p, s))

estimate_value(f::Function, mdp::Union{POMDP,MDP}, state, w::Float64, depth::Int) = f(mdp, state, w, depth)

"""
Construct a PISDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::PISDPWPlanner, s; tree_in_info=false, w=0.0, use_prior=true)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        tree = p.tree
        if !p.solver.keep_tree || tree == nothing
            tree = PISDPWTree{statetype(p.mdp),actiontype(p.mdp)}()
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
                put!(channel, Threads.@spawn simulate(p, snode, w, p.solver.depth, timeout_s; use_prior))
            end
        end

        nquery = 0
        for sim_task in sim_channel
            if timer() > timeout_s
                p.solver.show_progress ? finish!(progress) : nothing
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

        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        info[:search_time_us] = (timer() - start_s) * 1e6
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode, q_logprob = sample_sanode(tree, snode)
        a = sanode.a_label
        w_node = compute_IS_weight(q_logprob, a, use_prior ? actions(p.mdp, s) : nothing)
        w = w + w_node
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, w, info
end


"""
Return the reward for one iteration of MCTSDPW.
"""
function simulate(dpw::PISDPWPlanner, snode::PISDPWStateNode, w::Float64, d::Int, timeout_s::Float64=0.0; use_prior=true)
    sol = dpw.solver
    timer = sol.timer
    tree = dpw.tree
    s = snode.s_label
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0
    elseif d == 0 || (timeout_s > 0.0 && timer() > timeout_s)
        return estimate_value(dpw.solved_estimate, dpw.mdp, s, w, d)
    end

    # Action progressive widening.
    if sol.enable_action_pw
        if n_children(snode) <= sol.k_action * total_n(snode)^sol.alpha_action
            a = next_action(dpw.next_action, dpw.mdp, s, snode) # action generation step
            Base.@lock tree.state_action_nodes_lock begin
                insert_action_node!(tree, snode, a,
                                    init_N(sol.init_N, dpw.mdp, s, a),
                                    init_Q(sol.init_Q, dpw.mdp, s, a))
            end
        end
    elseif n_children(snode) == 0
        Base.@lock tree.state_action_nodes_lock begin
            for a in actions(dpw.mdp, s)
                insert_action_node!(tree, snode, a,
                                    init_N(sol.init_N, dpw.mdp, s, a),
                                    init_Q(sol.init_Q, dpw.mdp, s, a))
            end
        end
    end

    sanode, q_logprob = Base.@lock snode.s_lock begin; sample_sanode_UCB(tree, snode, sol.exploration_constant, sol.virtual_loss); end
    a = sanode.a_label
    w_node = compute_IS_weight(q_logprob, a, use_prior ? actions(dpw.mdp, s) : nothing) 
    w = w + w_node

    # State progressive widening.
    sp, r, spnode = nothing, nothing, nothing
    new_node = false
    Base.@lock sanode.a_lock begin
        if ((dpw.solver.enable_state_pw && n_a_children(sanode) <= sol.k_state * n(sanode)^sol.alpha_state) ||
            n_a_children(sanode) == 0)
            sp, r = @gen(:sp, :r)(dpw.mdp, s, [a, w], dpw.rng)

            spnode = Base.@lock tree.state_nodes_lock begin; insert_state_node!(tree, sp); end
            new_node = (n_children(spnode) == 0)
            push!(sanode.transitions, (sp, r))

            sanode.n_a_children += 1
            push!(sanode.unique_transitions, sp)
        else
            sp, r = rand(dpw.rng, sanode.transitions)
            spnode = Base.@lock tree.state_nodes_lock begin; tree.state_nodes[sp]; end
        end
    end

    if new_node
        q = r + discount(dpw.mdp) * estimate_value(dpw.solved_estimate, dpw.mdp, sp, w, d - 1)
    else
        q = r + discount(dpw.mdp) * simulate(dpw, spnode, w, d - 1)
    end

    function backpropagate(snode::PISDPWStateNode, sanode::PISDPWActionNode, q::Float64)
        snode.total_n += 1
        sanode.n += 1
        sanode.q += (q - sanode.q) / sanode.n
        delete!(snode.a_selected, sanode.a_label)
    end
    Base.@lock snode.s_lock begin; backpropagate(snode, sanode, q); end

    return q
end


function sample_sanode(tree::PISDPWTree, snode::PISDPWStateNode)
    all_Q = []
    all_sanodes = []
    for a_label in children(snode)
        state_action_key = (snode.s_label, a_label)
        sanode = tree.state_action_nodes[state_action_key]
        push!(all_Q, q(sanode))
        push!(all_sanodes, sanode)
    end
    sanode, q_logprob = select_action(all_sanodes, all_Q)
    return sanode, q_logprob
end


function sample_sanode_UCB(tree::PISDPWTree, snode::PISDPWStateNode, c::Float64, virtual_loss::Float64=0.0)
    all_UCB = []
    all_sanodes = []
    ltn = log(total_n(snode))
    for a_label in children(snode)
        state_action_key = (snode.s_label, a_label)
        sanode = tree.state_action_nodes[state_action_key]
        a_n = n(sanode)
        a_q = q(sanode)
        if (ltn <= 0 && a_n == 0) || c == 0.0
            UCB = a_q
        else
            UCB = a_q + c * sqrt(ltn / a_n)
        end

        vloss = (a_label in snode.a_selected ? virtual_loss : 0.0)
        UCB -= vloss

        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        
        push!(all_UCB, UCB)
        push!(all_sanodes, sanode)
    end
    sanode, q_logprob = select_action(all_sanodes, all_UCB)
    push!(snode.a_selected, sanode.a_label)
    return sanode, q_logprob
end
