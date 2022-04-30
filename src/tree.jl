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
    root::Union{Nothing,S}

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

    function PISTree{S,A}(root::Union{Nothing,S}=nothing) where {S,A}
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


function insert_action_node!(tree::PISTree{S,A}, snode::PISStateNode{S,A},
                             a::Union{S,A}, n0::Int, q0::Float64) where {S,A}
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
