using D3Trees
using Colors
using Printf

"""
Return text to display below the node corresponding to state or action s
"""
node_tag(s) = string(s)

"""
Return text to display in the tooltip for the node corresponding to state or action s
"""
tooltip_tag(s) = node_tag(s)


"""
Creates a D3Tree instance for a given tree.

Use it as
  a, info = action_info(planner, state)
  D3Tree(info[:tree])
"""
function D3Trees.D3Tree(tree::PISTree; title="MCTS-PIS Tree", kwargs...)
    lens = length(tree.state_nodes)
    lensa = length(tree.state_action_nodes)
    len = lens + lensa
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    style = fill("", len)
    link_style = fill("", len)
    max_q = maximum(node.q for (k, node) in tree.state_action_nodes)
    min_q = minimum(node.q for (k, node) in tree.state_action_nodes)

    snode_dict = Dict((snode.id, snode) for (_, snode) in tree.state_nodes)
    for (id, snode) in sort(collect(snode_dict), by=x->x[1])
        s_label = snode.s_label

        children_ids = sizehint!(Int64[], length(snode.children))
        for a in snode.children
            sanode = tree.state_action_nodes[(s_label, a)]
            push!(children_ids, sanode.id)

            w = 20.0 * sqrt(sanode.n / snode.total_n)
            link_style[sanode.id + lens] = "stroke-width:$(w)px"
        end

        children[id] = children_ids
        text[id] = Printf.@sprintf("""
                                   %25s
                                   N: %6d
                                   """,
                                   node_tag(s_label),
                                   snode.total_n)
        tt[id] = """
                 $(tooltip_tag(s_label))
                 N: $(snode.total_n)
                 """
    end
    sanode_dict = Dict((sanode.id, sanode) for (_, sanode) in tree.state_action_nodes)
    for (id, sanode) in sort(collect(sanode_dict), by=x->x[1])
        a_label = sanode.a_label

        children[id + lens] = let
            sp_children = if !isempty(sanode.unique_transitions)
                sanode.unique_transitions
            else
                first.(sanode.transitions)
            end
            @assert length(sp_children) == sanode.n_a_children
            collect(tree.state_nodes[sp].id for sp in sp_children)
        end

        text[id + lens] = @sprintf("""
                                   %25s
                                   Q: %6.2f
                                   N: %6d
                                   """,
                                   node_tag(a_label),
                                   sanode.q,
                                   sanode.n)
        tt[id + lens] = """
                        $(tooltip_tag(a_label))
                        Q: $(sanode.q)
                        N: $(sanode.n)
                        """

        rel_q = (sanode.q - min_q) / (max_q - min_q)
        if isnan(rel_q)
            color = colorant"gray"
        else
            color = weighted_color_mean(rel_q, colorant"green", colorant"red")
        end
        style[id + lens] = "stroke:#$(hex(color))"
    end

    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  style=style,
                  link_style=link_style,
                  title=title,
                  kwargs...)
end
