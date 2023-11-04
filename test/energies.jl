using Test: @test, @testset, @inferred
using Flux: flatten, Dense, Chain, relu
using Ab4Paper2023: GlobBias, IndepSite, Epistasis, ConstEnergy, ZeroEnergy,
    energies, SimpleAR, construct_ar_mask, DeepEnergy, AndEnergy
using OneHot: OneHotArray
using LinearAlgebra: dot
using LogExpFunctions: logsumexp

A = 4; L = 7; S = 5
seqs = rand(1:A, L, S)
seqs_ = OneHotArray(seqs, A) |> BitArray
@test OneHotArray(seqs, A) == seqs_

@testset "GlobBias" begin
    state = @inferred GlobBias(randn(A))
    @test @inferred(energies(seqs_, state)) ≈ -[sum(state.h[seqs[i,s]] for i=1:L) for s=1:S]
    @test eltype(GlobBias(A, Float32).h) == Float32
    @test eltype(GlobBias(A).h) == Float64
    @test size(GlobBias(A).h) == (A,)
end

@testset "IndepSite" begin
    state = @inferred IndepSite(randn(A,L))
    @test energies(seqs_, state) ≈ -[sum(state.h[seqs[i,s],i] for i=1:L) for s=1:S]
    @test eltype(IndepSite(A, L, Float32).h) == Float32
    @test eltype(IndepSite(A, L).h) == Float64
    @test size(IndepSite(A, L).h) == (A, L)
end

@testset "Epistasis" begin
    state = @inferred Epistasis(randn(A,L), randn(A,L,A,L))
    Eh = -[sum(state.h[seqs[i,s],i] for i=1:L) for s=1:S]
    EJ = -[sum(state.J[seqs[i,s],i,seqs[j,s],j] for i=1:L, j=1:L) for s=1:S]
    @test @inferred(energies(seqs_, state)) ≈ Eh + EJ
    @test eltype(Epistasis(A, L, Float32).h) == Float32
    @test eltype(Epistasis(A, L, Float32).J) == Float32
    @test eltype(Epistasis(A, L).h) == Float64
    @test eltype(Epistasis(A, L).J) == Float64
    @test size(Epistasis(A, L).h) == (A, L)
    @test size(Epistasis(A, L).J) == (A, L, A, L)
end

@testset "ConstEnergy" begin
    @test @inferred(energies(seqs_, ConstEnergy(2.3))) == fill(2.3, S)
    @test energies(seqs_, ConstEnergy(2.3)) isa Vector{Float64}
    @test ConstEnergy(2.3).e[] == 2.3
    @test eltype(ConstEnergy(2.3).e) == Float64
    @test eltype(ConstEnergy(Float32(2.3)).e) == Float32
    @test eltype(ConstEnergy(Float32).e) == Float32
    @test eltype(ConstEnergy().e) == Float64
end

@testset "ZeroEnergy" begin
    @test @inferred(energies(seqs_, ZeroEnergy())) == fill(0, S)
    @test ZeroEnergy(Float32) isa ZeroEnergy{Array{Float32,0}}
    @test ZeroEnergy() isa ZeroEnergy{Array{Bool,0}}
end

@testset "DeepEnergy" begin
    state = DeepEnergy(Chain(flatten, Dense(A*L, 1)))
    @test energies(seqs_, state) == vec(state.m.layers[2].weight * reshape(seqs_, A * L, :) .+ state.m.layers[2].bias)
    state = DeepEnergy(Chain(flatten, Dense(A*L, 5, relu), Dense(5, 1)))
    @test energies(seqs_, state) == vec(state.m(seqs_))
    @test size(energies(seqs_, state)) == (S,)
end

@testset "SimpleAR" begin
    mask = [i > j for a=1:A, i=1:L, b=1:A, j=1:L]
    @test construct_ar_mask(A, L) == mask
    state = SimpleAR(A, L)
    @test state.mask == mask
    E = energies(seqs_, state)
    for s = 1:S
        logP = 0.0
        for i = 1:L
            logP += dot(seqs_[:,i,s], state.biases[:,i]) + sum(dot(seqs_[:,i,s], state.weights[:,i,:,j], seqs_[:,j,s]) for j=1:i-1; init=0.0)
            logP -= logsumexp([state.biases[a,i] + sum(dot(state.weights[a,i,:,j], seqs_[:,j,s]) for j = 1:i-1; init=0.0) for a=1:A])
        end
        @test logP ≈ E[s]
    end
end

@testset "AndEnergy" begin
    and = @inferred AndEnergy(IndepSite(randn(A,L)), IndepSite(randn(A,L)))
    @test @inferred(energies(seqs_, and)) ≈ energies(seqs_, and.s[1]) + energies(seqs_, and.s[2])
end
