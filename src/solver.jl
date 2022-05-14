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
    nominal_distrib_fn::Function
        A function that takes in an MDP and a state as input and returns the
        nominal distribution over actions. Used for computing the weights.
    cost_reduction::Symbol
    action_selection::Symbol
        The type of strategy to use for reducing costs, selecting actions, etc.
    experiment_config::Any
        Various experimental settings. After finishing experimenting should be
        either removed or promoted to be separat ctor args. See the struct for
        the current experiment flags and definitions.
    keep_tree::Bool
        If true, store the tree in the planner for reuse at the next timestep.
        default: false
    enable_action_pw::Bool
        If true, enable progressive widening on the action space; if false just use the whole action space.
        default: true
    enable_state_pw::Bool
        If true, enable progressive widening on the state space; if false just use the single next state (for deterministic problems).
        default: true
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
    nominal_distrib_fn::Function
    cost_reduction::Symbol
    action_selection::Symbol
    experiment_config::Any
    keep_tree::Bool
    enable_action_pw::Bool
    enable_state_pw::Bool
    rng::AbstractRNG
    init_Q::Any
    init_N::Any
    next_action::Any
    default_action::Any
    reset_callback::Function
    show_progress::Bool
    timer::Function
end

Base.@kwdef struct ExperimentConfig
    nominal_steps::Int = 0
    update_period::Int = -1
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
                   nominal_distrib_fn=(mdp, s)->Normal(0, 1),
                   cost_reduction::Symbol=:sum,
                   action_selection::Symbol=:adaptive,
                   experiment_config::Any=ExperimentConfig(),
                   keep_tree::Bool=false,
                   enable_action_pw::Bool=true,
                   enable_state_pw::Bool=true,
                   rng::AbstractRNG=Random.GLOBAL_RNG,
                   init_Q::Any=0.0,
                   init_N::Any=1,
                   next_action::Any=UniformActionGenerator(rng),
                   default_action::Any=nothing,
                   reset_callback::Function=(mdp, s)->false,
                   show_progress::Bool=false,
                   timer=()->1e-9*time_ns())
    PISSolver(depth, exploration_constant, n_iterations, max_time, k_action,
              alpha_action, k_state, alpha_state, virtual_loss,
              nominal_distrib_fn, cost_reduction, action_selection,
              experiment_config, keep_tree, enable_action_pw, enable_state_pw,
              rng, init_Q, init_N, next_action, default_action, reset_callback,
              show_progress, timer)
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
                      # TODO(kykim): Temporary hack to get around the RMDP case.
                      # Should be actiontype(P).
                      Any,
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

POMDPs.solve(solver::PISSolver, mdp::Union{POMDP,MDP}) = PISPlanner(solver, mdp)
