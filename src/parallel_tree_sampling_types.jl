"""
MCTS solver with Importance Sampling

Fields:
    depth::Int64
        Maximum rollout horizon and tree depth.
        default: 10
    exploration_constant::Float64
        Specified how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0
    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100
    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf
    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
        These constants control the double progressive widening. A new state
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k:10, alpha:0.5
    virtual_loss::Float64
        A temporary loss added to the UCB score for nodes that are currently being
        evaluated by some threads. This can be used to encourage threads to explore
        broader parts of the search space. Relevant when running MCTS with multiple threads.
        default: 0.0
    keep_tree::Bool
        If true, store the tree in the planner for reuse at the next timestep (and every time it is used in the future). There is a computational cost for maintaining the state dictionary necessary for this.
        default: false
    enable_action_pw::Bool
        If true, enable progressive widening on the action space; if false just use the whole action space.
        default: true
    enable_state_pw::Bool
        If true, enable progressive widening on the state space; if false just use the single next state (for deterministic problems).
        default: true
    check_repeat_state::Bool
    check_repeat_action::Bool
        When constructing the tree, check whether a state or action has been seen before (there is a computational cost to maintaining the dictionaries necessary for this)
        default: true
    tree_in_info::Bool
        If true, return the tree in the info dict when action_info is called. False by default because it can use a lot of memory if histories are being saved.
        default: false
    rng::AbstractRNG
        Random number generator
    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will always be set to that number.
        default: 0.0
    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will always be set to that number.
        default: 0
    next_action::Any
        Function or object used to choose the next action to be considered for progressive widening.
        The next action is determined based on the MDP, the state, `s`, and the current `DPWStateNode`, `snode`.
        If this is a function `f`, `f(mdp, s, snode)` will be called to set the value.
        If this is an object `o`, `next_action(o, mdp, s, snode)` will be called.
        default: RandomActionGenerator(rng)
    default_action::Any
        Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
        If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
        If this is a Policy `p`, `action(p, belief)` will be called.
        If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.
        default: `ExceptionRethrow()`
    reset_callback::Function
        Function used to reset/reinitialize the MDP to a given state `s`.
        Useful when the simulator state is not truly separate from the MDP state.
        `f(mdp, s)` will be called.
        default: `(mdp, s)->false` (optimized out)
    show_progress::Bool
        Show progress bar during simulation.
        default: false
"""
# TODO(kykim): Clean up some of the settings.
mutable struct PISSolver
    depth::Int
    exploration_constant::Float64
    n_iterations::Int
    max_time::Float64
    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
    virtual_loss::Float64
    keep_tree::Bool
    enable_action_pw::Bool
    enable_state_pw::Bool
    check_repeat_state::Bool
    check_repeat_action::Bool
    tree_in_info::Bool
    rng::AbstractRNG
    init_Q::Any
    init_N::Any
    next_action::Any
    default_action::Any
    reset_callback::Function
    show_progress::Bool
    timer::Function
    α::Float64
end

mutable struct UniformActionGenerator{RNG<:AbstractRNG}
    rng::RNG
end
UniformActionGenerator() = UniformActionGenerator(Random.GLOBAL_RNG)

"""
Use keyword arguments to specify values for the fields.
"""
function PISSolver(;depth::Int=10,
                   exploration_constant::Float64=1.0,
                   n_iterations::Int=100,
                   max_time::Float64=Inf,
                   k_action::Float64=10.0,
                   alpha_action::Float64=0.5,
                   k_state::Float64=10.0,
                   alpha_state::Float64=0.5,
                   virtual_loss::Float64=0.0,
                   keep_tree::Bool=false,
                   enable_action_pw::Bool=true,
                   enable_state_pw::Bool=true,
                   check_repeat_state::Bool=true,  # TODO(kykim): Are the two needed?
                   check_repeat_action::Bool=true,
                   tree_in_info::Bool=false,
                   rng::AbstractRNG=Random.GLOBAL_RNG,
                   init_Q::Any=0.0,
                   init_N::Any=1,
                   next_action::Any=UniformActionGenerator(rng),
                   default_action::Any=nothing,
                   reset_callback::Function=(mdp, s) -> false,
                   show_progress::Bool=false,
                   timer=() -> 1e-9*time_ns(),
                   α::Float64=0.1)
    PISSolver(depth, exploration_constant, n_iterations, max_time, k_action, alpha_action, k_state, alpha_state, virtual_loss,
              keep_tree, enable_action_pw, enable_state_pw, check_repeat_state, check_repeat_action, tree_in_info, rng,
              init_Q, init_N, next_action, default_action, reset_callback, show_progress, timer, α)
end


# TODO(kykim): Try a lock-free approach using Atomics.
mutable struct PISActionNode{S,A}
    id::Int
    a_label::A
    n::Int
    q::Float64
    transitions::Vector{Tuple{S,Float64}}
    unique_transitions::Set{S}
    n_a_children::Int
    conditional_cdf_est::RunningCDFEstimator
    a_lock::ReentrantLock
end
PISActionNode(id::Int, a::A, n::Int, q::Float64, transitions::Vector{Tuple{S,Float64}}) where {S,A} =
    PISActionNode{S,A}(id, a, n, q, transitions, Set{S}(), 0, RunningCDFEstimator([0.0], [1e-5]), ReentrantLock())

@inline n(n::PISActionNode) = n.n
@inline q(n::PISActionNode) = n.q
@inline n_a_children(n::PISActionNode) = n.n_a_children


mutable struct PISStateNode{S,A}
    id::Int
    s_label::S
    total_n::Int
    children::Vector{A}
    s_lock::ReentrantLock
    # Action nodes currently being evaluated. Used for applying virtual loss.
    a_selected::Set{A}
end
PISStateNode(id::Int, s::S, total_n::Int, children::Vector{A}) where {S,A} =
    PISStateNode{S,A}(id, s, total_n, children, ReentrantLock(), Set{A}())

@inline total_n(n::PISStateNode) = n.total_n
@inline children(n::PISStateNode) = n.children
@inline n_children(n::PISStateNode) = length(children(n))
@inline isroot(n::PISStateNode) = (n.id == 1)


mutable struct PISTree{S,A}
    root::Union{Nothing, S}

    state_nodes::Dict{S, PISStateNode}
    state_action_nodes::Dict{Tuple{S,A}, PISActionNode}

    cdf_est::RunningCDFEstimator
    costs::Vector{Float64}
    weights::Vector{Float64}

    state_nodes_lock::ReentrantLock
    state_action_nodes_lock::ReentrantLock
    costs_weights_lock::ReentrantLock

    # Used to assign unique ids to nodes.
    _s_id_counter::Threads.Atomic{Int}
    _a_id_counter::Threads.Atomic{Int}

    function PISTree{S,A}(root::Union{Nothing, S}=nothing) where {S,A} 
        return new(root,
                   Dict{S, PISStateNode}(),
                   Dict{Tuple{S,A}, PISActionNode}(),

                   RunningCDFEstimator([0.0], [1e-7]),
                   sizehint!(Float64[], 10_000),
                   sizehint!(Float64[], 10_000),

                   ReentrantLock(),
                   ReentrantLock(),
                   ReentrantLock(),

                   Threads.Atomic{Int}(1),
                   Threads.Atomic{Int}(1))
    end
end

Base.isempty(tree::PISTree) = isempty(tree.state_nodes)


function insert_state_node!(tree::PISTree{S,A}, s::S) where {S,A}
    snode = nothing
    Base.@lock tree.state_nodes_lock begin
        haskey(tree.state_nodes, s) && return tree.state_nodes[s]
        id = Threads.atomic_add!(tree._s_id_counter, 1)
        snode = PISStateNode(id, s, 0, Vector{A}())
        tree.state_nodes[s] = snode
    end
    return snode
end


function insert_action_node!(tree::PISTree{S,A}, snode::PISStateNode{S,A}, a::A, n0::Int, q0::Float64) where {S,A}
    sanode = nothing
    Base.@lock tree.state_action_nodes_lock begin
        state_action_key = (snode.s_label, a)
        haskey(tree.state_action_nodes, state_action_key) && return tree.state_action_nodes[state_action_key]
        id = Threads.atomic_add!(tree._a_id_counter, 1)
        sanode = PISActionNode(id, a, n0, q0, Vector{Tuple{S,Float64}}())
        tree.state_action_nodes[state_action_key] = sanode
    end
    Base.@lock snode.s_lock begin
        push!(snode.children, a)
        snode.total_n += n0
    end
    return sanode
end


mutable struct PISPlanner{P<:Union{MDP,POMDP}, S, A, NA, RCB, RNG}
    solver::PISSolver
    mdp::P
    tree::Union{Nothing, PISTree{S,A}}
    next_action::NA
    reset_callback::RCB
    rng::RNG
end


function PISPlanner(solver::PISSolver, mdp::P) where P<:Union{POMDP,MDP}
    return PISPlanner{P,
                      statetype(P),
                      actiontype(P),
                      typeof(solver.next_action),
                      typeof(solver.reset_callback),
                      typeof(solver.rng)}(
                          solver,
                          mdp,
                          nothing,
                          solver.next_action,
                          solver.reset_callback,
                          solver.rng)
end

Random.seed!(p::PISPlanner, seed) = Random.seed!(p.rng, seed)
