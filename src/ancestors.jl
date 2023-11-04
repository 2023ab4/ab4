#=
We can represent a rooted tree as a list of ancestors.
So that ancestors[i] gives the ancestor of node 'i'.
If 'i' is a root node, ancestors[i] == 0.
We assume that 'ancestors' always visits parents before children.

Edges are identified by the index of the descendant node.

We can also represent forests of rooted trees.
=#

function valid_ancestors(ancestors)
    for (i, a) in enumerate(ancestors)
        0 ≤ a < i || return false # children are traversed before parents?
    end
    return true
end

number_of_nodes(ancestors) = length(ancestors)
number_of_roots(ancestors) = count(iszero, ancestors)
number_of_edges(ancestors) = number_of_nodes(ancestors) - number_of_roots(ancestors)
isroot(i::Int, ancestors) = iszero(ancestors[i])
isleaf(i::Int, ancestors) = (i ∉ ancestors)

"""
    find_root(node, ancestors)

Returns the root of the tree containing `node`.
"""
function find_root(node::Int, ancestors)
    @assert valid_ancestors(ancestors)
    @assert 1 ≤ node ≤ length(ancestors)
    a = node
    while a > 0
        node = a
        a = ancestors[node]
    end
    return node
end

"""
    node_costs(X, ancestors)

Given edge costs `X`, returns costs associated to traversals from roots to nodes.
"""
function node_costs(X::AbstractArray, ancestors; dim::Int = ndims(X))
    @assert size(X, dim) == number_of_edges(ancestors)
    Y_sz = ntuple(d -> d == dim ? number_of_nodes(ancestors) : size(X, d), Val(ndims(X)))
    Y = fill!(similar(X, typeof(sum(X)), Y_sz), 0)
    edge_index = 0
    for (i, a) in enumerate(ancestors)
        if a > 0
            edge_cost = selectdim(X, dim, edge_index += 1)
            selectdim(Y, dim, i) .= selectdim(Y, dim, a) .+ edge_cost
        end
    end
    return Y
end

# since this involves mutation and loops, we define the rrule manually
function ChainRulesCore.rrule(
    ::typeof(node_costs),
    X::AbstractArray,
    ancestors;
    dim::Int = ndims(X)
)
    Y = node_costs(X, ancestors; dim = dim)
    function node_costs_pullback(dY)
        @assert size(dY) == size(Y)
        dX = zero(X)
        n_roots = cumsum(iszero.(ancestors))
        @assert issorted([i - n_roots[i] for i in eachindex(ancestors)])
        for (i, a) in Iterators.reverse(enumerate(ancestors))
            if a > 0
                i_ = i - n_roots[i]
                a_ = a - n_roots[a]
                selectdim(dX, dim, i_) .+= selectdim(dY, dim, i)
                if ancestors[a] > 0
                    selectdim(dX, dim, a_) .+= selectdim(dX, dim, i_)
                end
            end
        end
        return NoTangent(), dX, NoTangent()
    end
    return Y, node_costs_pullback
end

"""
    edge_costs(X, ancestors)

Given node costs `Y`, returns the costs associated to edges.
"""
function edge_costs(Y::AbstractArray, ancestors; dim::Int = ndims(Y))
    @assert size(Y, dim) == number_of_nodes(ancestors)
    X_sz = ntuple(d -> d == dim ? number_of_edges(ancestors) : size(Y, d), Val(ndims(Y)))
    X = fill!(similar(Y, eltype(Y), X_sz), 0)
    r = 0
    for (i, a) in enumerate(ancestors)
        if a > 0
            r += 1
            selectdim(X, dim, r) .= selectdim(Y, dim, i) .- selectdim(Y, dim, a)
        end
    end
    return X
end

"""
    subtree_sum(X, ancestors)

Given node costs `X`, computes the total cost of the subtree subyacent at each node.
"""
function subtree_sum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    @assert size(X, dim) == number_of_nodes(ancestors)
    Y = copy(X)
    for (i, a) in Iterators.reverse(enumerate(ancestors))
        if a > 0
            selectdim(Y, dim, a) .+= selectdim(Y, dim, i)
        end
    end
    return Y
end

"""
    tree_sum(X, ancestors)

Given node costs `X`, assigs to each node the total cost of the tree that contains it.
"""
function tree_sum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    Y = subtree_sum(X, ancestors; dim = dim)
    for (i, a) in enumerate(ancestors)
        if a > 0
            selectdim(Y, dim, i) .= selectdim(Y, dim, a)
        end
    end
    return Y
end

function ChainRulesCore.rrule(
    ::typeof(tree_sum),
    X::AbstractArray,
    ancestors;
    dim::Int = ndims(X)
)
    Y = tree_sum(X, ancestors; dim = dim)
    function tree_sum_pullback(dY)
        dX = tree_sum(dY, ancestors; dim = dim)
        return NoTangent(), dX, NoTangent()
    end
    return Y, tree_sum_pullback
end

"""
    tree_logsumexp(X, ancestors; dim = ndims(X))

Assigns the value of log(sum(exp.(X) for each tree)) to each node.
"""
function tree_logsumexp(X::AbstractArray, ancestors; dim::Int = ndims(X))
    X_max = tree_maximum(X, ancestors; dim = dim)
    return log.(tree_sum(exp.(X .- X_max), ancestors; dim = dim)) .+ X_max
end

function ChainRulesCore.rrule(
    ::typeof(tree_logsumexp),
    X::AbstractArray,
    ancestors;
    dim::Int = ndims(X)
)
    Y = tree_logsumexp(X, ancestors; dim = dim)
    function tree_logsumexp_pullback(dY)
        dX = tree_sum(dY, ancestors; dim = dim) .* exp.(X .- Y)
        return NoTangent(), dX, NoTangent()
    end
    return Y, tree_logsumexp_pullback
end

"""
    subtree_maximum(X, ancestors; dim = ndims(X))

Assigns to each node the maximum cost of the nodes at the subyacent subtree.
"""
function subtree_maximum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    Y = copy(X)
    for (i, a) in Iterators.reverse(enumerate(ancestors))
        if a > 0
            selectdim(Y, dim, a) .= max.(selectdim(Y, dim, i), selectdim(Y, dim, a))
        end
    end
    return Y
end

"""
    tree_maximum(X, ancestors; dim = ndims(X))

Assigns to each node the maximum cost of all nodes at the tree that contains it.
"""
function tree_maximum(X::AbstractArray, ancestors; dim::Int = ndims(X))
    Y = subtree_maximum(X, ancestors; dim = dim)
    for (i, a) in enumerate(ancestors)
        if a > 0
            selectdim(Y, dim, i) .= selectdim(Y, dim, a)
        end
    end
    return Y
end
