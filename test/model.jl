import Flux
using Ab4Paper2023: ConstEnergy
using Ab4Paper2023: Data
using Ab4Paper2023: depletion_gradient!
using Ab4Paper2023: energies
using Ab4Paper2023: Epistasis
using Ab4Paper2023: IndepModel
using Ab4Paper2023: IndepSite
using Ab4Paper2023: log_abundances
using Ab4Paper2023: log_likelihood
using Ab4Paper2023: log_likelihood_samples
using Ab4Paper2023: log_multinomial
using Ab4Paper2023: log_selectivities
using Ab4Paper2023: Model
using Ab4Paper2023: number_of_rounds
using Ab4Paper2023: number_of_sequences
using Ab4Paper2023: rare_binding_gauge
using Ab4Paper2023: rare_binding_gauge_zeta
using Ab4Paper2023: rare_binding_gauge_zeta!
using Ab4Paper2023: rare_binding_gauge!
using Ab4Paper2023: select_sequences
using Ab4Paper2023: sum_
using Ab4Paper2023: unsqueeze_left
using Ab4Paper2023: valid_ancestors
using Ab4Paper2023: ZeroEnergy
using Ab4Paper2023: zerosum
using FiniteDifferences: central_fdm
using FiniteDifferences: grad
using LinearAlgebra: norm
using Random: randn!
using Statistics: mean
using Test: @test
using Test: @testset
using Zygote: gradient

function random_data(; A::Int = 3, L::Int = 4, T::Int = 7, W::Int = 4, S::Int = 128)
    sequences = falses(A, L, S)
    for s=1:S, i=1:L
        sequences[rand(1:A), i, s] = true
    end
    ancestors = random_ancestors(T)
    counts = rand(S, T)
    return Data(sequences, counts, ancestors)
end

function random_ancestors(n::Int)
    ancestors = ntuple(i -> rand(0:(i - 1)), n)
    @assert valid_ancestors(ancestors)
    return ancestors
end

function random_select(W::Int, R::Int)
    select = falses(W, R)
    for t in 1:R
        select[rand(1:W), t] = true
    end
    washed = (!).(select)
    return select, washed
end

@testset "select_sequences" begin
    data = random_data()
    selseqflag = rand(Bool, number_of_sequences(data))
    selseqdata = select_sequences(data, selseqflag)
    @test selseqdata.sequences == data.sequences[:,:,selseqflag]
    @test selseqdata.counts == data.counts[selseqflag,:]
    @test selseqdata.ancestors == data.ancestors
end

@testset "log_likelihood" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    W = 4
    select, washed = random_select(W, R)
    states = (IndepSite(A,L), Epistasis(A,L), ConstEnergy(fill(randn())), ZeroEnergy())
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    lp = log_selectivities(model, data)
    lM = dropdims(log_multinomial(data.counts; dims=1); dims=1)
    Ls = lM .+ sum_(data.counts .* log_abundances(model, data); dims=1)

    @test log_likelihood(lp, model.ζ, data) ≈ log_likelihood(model, data)
    @test energies(data, model) == energies(data.sequences, model)
    @test log_selectivities(model, data; rare_binding=false) == log_selectivities(model, data)
    @test all(log_selectivities(model, data; rare_binding=false) .≤ log_selectivities(model, data; rare_binding=true))
    @test log_likelihood_samples(model, data) ≈ Ls / number_of_sequences(data)
    @test log_likelihood(model, data) ≈ sum(Ls) / number_of_sequences(data)
    @test log_likelihood(model, data) ≈ sum(log_likelihood_samples(model, data))

    function fun(x)
        model_ = Model(
            (IndepSite(x), states[2:end]...),
            model.μ, model.ζ, model.select, model.washed
        )
        return log_likelihood(model_, data)
    end

    ps = Flux.params(model)
    gs = gradient(ps) do
        log_likelihood(model, data)
    end
    @test gs[states[1].h] ≈ grad(central_fdm(5, 1), fun, model.states[1].h)[1]

    function gun(J)
        model_ = Model(
            (states[1], Epistasis(states[2].h, J), states[3:end]...),
            model.μ, model.ζ, model.select, model.washed
        )
        return log_likelihood(model_, data)
    end

    ps = Flux.params(model)
    gs = gradient(ps) do
        log_likelihood(model, data)
    end
    @test gs[states[2].J] ≈ grad(central_fdm(5, 1), gun, states[2].J)[1]
end

@testset "depletion_gradient" begin
    data = random_data()
    lp = randn(number_of_sequences(data), number_of_rounds(data))
    ζ = randn(number_of_rounds(data))
    lN = log_abundances(lp, ζ, data)
    gs = gradient(Flux.params(ζ)) do
        log_likelihood(lp, ζ, data)
    end
    G = depletion_gradient!(zero(ζ), lN, data)
    @test G ≈ -gs[ζ]
end

@testset "rare binding gauge" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), IndepSite(A,L), IndepSite(A,L), IndepSite(A,L))
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    model_ = rare_binding_gauge(model)
    @test norm(sum(model_.μ .* model_.select; dims=1)) ≤ 1e-10
    @test norm(sum(model_.μ .* model_.washed; dims=1)) ≤ 1e-10
    @test model_.states == model.states
    lpz  = log_selectivities(model,  data; rare_binding=true) .- reshape(model.ζ,  1, :)
    lpz_ = log_selectivities(model_, data; rare_binding=true) .- reshape(model_.ζ, 1, :)
    @test lpz_ ≈ lpz

    model__ = deepcopy(model)
    rare_binding_gauge!(model__)
    @test model_.μ == model__.μ
    @test model_.ζ == model__.ζ
    for w = 1:W
        @test model_.states[w].h == model__.states[w].h
    end
end

@testset "rare_binding_gauge_zeta" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), IndepSite(A,L), IndepSite(A,L), IndepSite(A,L))
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    model_ = rare_binding_gauge_zeta(model)
    @test iszero(model_.ζ)
    @test model_.states == model.states
    lpz  = log_selectivities(model,  data; rare_binding=true) .- reshape(model.ζ,  1, :)
    lpz_ = log_selectivities(model_, data; rare_binding=true) .- reshape(model_.ζ, 1, :)
    @test lpz_ ≈ lpz

    model__ = deepcopy(model)
    rare_binding_gauge_zeta!(model__)
    @test model_.μ == model__.μ
    @test model_.ζ == model__.ζ
    for w = 1:W
        @test model_.states[w].h == model__.states[w].h
    end
end

@testset "zerosum" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), IndepSite(A,L), IndepSite(A,L), IndepSite(A,L))
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    indep_model = IndepModel(model)
    indep_model_ = zerosum(indep_model)
    @test norm(mean(indep_model_.states.h; dims=1)) ≤ 1e-10
    @test indep_model.ζ == indep_model_.ζ
    @test indep_model.select == indep_model_.select
    @test indep_model.washed == indep_model_.washed
    e0 = energies(data, indep_model)  .- unsqueeze_left(indep_model.μ)
    e1 = energies(data, indep_model_) .- unsqueeze_left(indep_model_.μ)
    @test e0 ≈ e1
end
