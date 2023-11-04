using Ab4Paper2023: experiment_with_targets, Data
using AbstractTrees: PreOrderDFS
using Test: @test, @testset

@testset "experiment_with_targets, both" begin
    data = Data(experiment_with_targets(; colors=["both"]))
    @test size(data.sequences) == (20, 4, 76924)
end

@testset "experiment_with_targets, subbranch" begin
    root = experiment_with_targets(; colors=["blue"])
    node_idx = Dict(n.label => t for (t, n) in enumerate(PreOrderDFS(root)))
    data = Data(root)
    for n = PreOrderDFS(root)
        if occursin("black", n.label) || occursin("both", n.label)
            @test iszero(data.counts[:, node_idx[n.label]])
        elseif occursin("blue", n.label)
            @test !iszero(data.counts[:, node_idx[n.label]])
        else
            @test n.label == "root"
            @test !iszero(data.counts[:, node_idx[n.label]])
        end
    end
end
