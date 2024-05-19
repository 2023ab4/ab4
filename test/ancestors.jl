using Ab4Paper2023: edge_costs
using Ab4Paper2023: node_costs
using Ab4Paper2023: find_root
using Ab4Paper2023: isleaf
using Ab4Paper2023: isroot
using Ab4Paper2023: number_of_edges
using Ab4Paper2023: number_of_nodes
using Ab4Paper2023: valid_ancestors
using Ab4Paper2023: subtree_sum
using Ab4Paper2023: tree_sum
using Ab4Paper2023: subtree_maximum
using Ab4Paper2023: tree_maximum
using Ab4Paper2023: tree_logsumexp
using ChainRulesCore: rrule
using FiniteDifferences: central_fdm
using FiniteDifferences: grad
using Test: @test
using Test: @testset

"""
    random_ancestors(n)

Creates a random ancestors list of length `n`.
"""
function random_ancestors(n::Int)
    ancestors = ntuple(i -> rand(0:(i - 1)), n)
    @assert valid_ancestors(ancestors)
    return ancestors
end

@testset "ancestors" begin
    for _ in 1:10
        ancestors = random_ancestors(rand(10:20))
        @test number_of_edges(ancestors) == count(!iszero, ancestors)
        roots = [find_root(i, ancestors) for i in eachindex(ancestors)]
        for (i, a) in enumerate(ancestors)
            @test isroot(i, ancestors) == iszero(a)
            @test !isleaf(a, ancestors)
            @test isleaf(i, ancestors) == (i ∉ ancestors)
            if iszero(a)
                @test roots[i] == i
            else
                @test roots[i] == roots[a]
            end
        end
    end
end

@testset "node_costs" begin
    for _ in 1:10
        ancestors = random_ancestors(rand(10:20))
        edge_indices = cumsum((!iszero).(ancestors))
        X = randn(rand(3:7), number_of_edges(ancestors))
        Y = zeros(size(X, 1), number_of_nodes(ancestors))
        for (i, a) in enumerate(ancestors)
            if a > 0
                Y[:, i] = Y[:, a] + X[:, edge_indices[i]]
            end
        end
        @test node_costs(X, ancestors) ≈ Y
        @test edge_costs(Y, ancestors) ≈ X

        G = grad(central_fdm(5,1), x -> sum(node_costs(x, ancestors)), X)[1]
        _, pb = rrule(node_costs, X, ancestors)
        @test pb(ones(size(Y)...))[2] ≈ G
    end
end

@testset "subtree_sum" begin
    for _ in 1:10
        ancestors = random_ancestors(rand(10:20))
        X = randn(rand(2:5), rand(2:4), number_of_nodes(ancestors))
        Y = subtree_sum(X, ancestors)
        for (i, a) in enumerate(ancestors)
            if isleaf(i, ancestors)
                @test selectdim(Y, ndims(Y), i) ≈ selectdim(X, ndims(X), i)
            end
            if a > 0
                child_costs = sum(selectdim(Y, ndims(Y), j) for (j, b) in enumerate(ancestors) if a == b)
                c = selectdim(X, ndims(X), a) + child_costs
                @test selectdim(Y, ndims(Y), a) ≈ c
            end
        end
    end
end

@testset "tree_sum" begin
    for _ in 1:10
        ancestors = random_ancestors(rand(10:20))
        X = randn(rand(2:5), rand(2:4), number_of_nodes(ancestors))
        Y = tree_sum(X, ancestors)
        Z = subtree_sum(X, ancestors)
        for (i, a) in enumerate(ancestors)
            if a > 0
                @test selectdim(Y, ndims(Y), i) ≈ selectdim(Y, ndims(Y), a)
            else
                @test selectdim(Y, ndims(Y), i) ≈ selectdim(Z, ndims(Z), i)
            end
        end
        G = grad(central_fdm(5,1), x -> sum(tree_sum(x, ancestors)), X)[1]
        _, pb = rrule(tree_sum, X, ancestors)
        @test pb(ones(size(Y)...))[2] ≈ G
    end
end

@testset "subtree_maximum" begin
    for _ in 1:10
        ancestors = random_ancestors(rand(10:20))
        X = randn(rand(2:5), rand(2:4), number_of_nodes(ancestors))
        Y = subtree_maximum(X, ancestors)
        for (i, a) in enumerate(ancestors)
            if a > 0
                @test all(selectdim(Y, ndims(Y), i) .≤ selectdim(Y, ndims(Y), a))
                children = [selectdim(Y, ndims(Y), j) for (j, b) in enumerate(ancestors) if a == b]
                @test all(selectdim(Y, ndims(Y), a) .== max.(selectdim(X, ndims(X), a), children...))
            end
            if isleaf(i, ancestors)
                @test selectdim(Y, ndims(Y), i) == selectdim(X, ndims(X), i)
            end
        end
    end
end

@testset "tree_maximum" begin
    for _ in 1:10
        ancestors = random_ancestors(rand(10:20))
        X = randn(rand(2:5), rand(2:4), number_of_nodes(ancestors))
        Y = tree_maximum(X, ancestors)
        Z = subtree_maximum(X, ancestors)
        for (i, a) in enumerate(ancestors)
            if a > 0
                @test all(selectdim(Y, ndims(Y), i) .== selectdim(Y, ndims(Y), a))
            else
                @test all(selectdim(Y, ndims(Y), i) .== selectdim(Z, ndims(Z), i))
            end
        end
    end
end

@testset "tree_logsumexp" begin
    for _ in 1:1
        ancestors = random_ancestors(rand(7:15))
        X = rand(rand(2:3), rand(2:3), number_of_nodes(ancestors)) .- 0.5
        Y = tree_logsumexp(X, ancestors)
        @test Y ≈ log.(tree_sum(exp.(X), ancestors))
        G = grad(central_fdm(5,1), x -> sum(tree_logsumexp(x, ancestors)), X)[1]
        _, pb = rrule(tree_logsumexp, X, ancestors)
        @test pb(ones(size(Y)...))[2] ≈ G
    end
end
