struct Model{St,M,Z,F}
    states::St
    μ::M # μ[w,r] chemical potential of state 'w' in round 'r'
    ζ::Z # exp(-ζ[r]) is the amplification factor at round 'r'
    select::F # select[w,r] is true if state 'w' is selected in round 'r'
    washed::F # washed[w,r] is true if state 'w' is washed out in round 'r'
    function Model(
        states,
        μ::AbstractMatrix,
        ζ::AbstractVector,
        select::AbstractMatrix,
        washed::AbstractMatrix,
    )
        @assert size(μ, 1) == number_of_states(states)
        @assert size(μ, 2) == length(ζ)
        @assert size(μ) == size(select) == size(washed)
        @assert all((select .== 0) .| (select .== 1))
        @assert all((washed .== 0) .| (washed .== 1))
        @assert iszero(select .& washed) # select and washed states are disjoint
        @assert all(any(select; dims = 1)) # at least one state selected in every round
        return new{typeof(states), typeof(μ), typeof(ζ), typeof(select)}(states, μ, ζ, select, washed)
    end
end
Flux.trainable(model::Model) = (model.states, model.μ, model.ζ)
@functor Model

"""
    number_of_rounds(model)

Number of selection rounds modelled in `model`.
"""
number_of_rounds(model::Model) = length(model.ζ)
number_of_states(model::Model) = length(model.states)
number_of_states(states::Tuple) = length(states)

"""
    log_abundances(model, data; rare_binding = false, batch = MiniBatch(data))

Estimates of abundances, log(N[s,t]), for sequences in 'batch'.
"""
function log_abundances(
    model::Model,
    data::Data;
    rare_binding::Bool = false,
    batch::MiniBatch = MiniBatch(data)
)
    lp = log_selectivities(model, batch; rare_binding = rare_binding) # lp[s,r]
    return log_abundances(lp, model.ζ, data; batch = batch)
end

function log_abundances(
    lp::AbstractMatrix,
    ζ::AbstractVector,
    data::Data;
    batch::MiniBatch = MiniBatch(data)
)
    lPZ = node_costs(lp .- reshape(ζ, 1, :), data.ancestors)
    return batch.lRs .+ lPZ .- tree_logsumexp(lPZ .+ data.lRt, data.ancestors; dim = 2)
end

function log_selectivities(model::Model, data::Union{Data, MiniBatch}; kwargs...)
    return log_selectivities(model, data.sequences; kwargs...)
end

function log_selectivities(
    model::Model,
    sequences::Sequences;
    rare_binding::Bool = false
)
    E = energies(sequences, model)
    μ_num = potentials_select(model.μ; select = model.select)
    if rare_binding
        μ_den = potentials_select(model.μ; select = model.washed)
    else
        μ_den = potentials_select(model.μ; select = model.washed .| model.select)
    end
    # Eeff[s,w,r]
    Eeff_num = effective_energies(E, μ_num)
    Eeff_den = effective_energies(E, μ_den)
    return logsumexp_(-Eeff_num; dims = 2) - logsumexp_(-Eeff_den; dims = 2) # lp[s,r]
end

"""
    effective_energies(E, μ)

Tensor of effective energies, Eeff[s,w,r] = E[s,w] - μ[w,r].
"""
function effective_energies(E::AbstractMatrix, μ::AbstractMatrix)
    return unsqueeze_right(E) .- unsqueeze_left(μ)
end

"""
    potentials_select(μ; select)

Assigns `-Inf` chemical potential to non-selected states. In other words, replaces
entries in `μ` with `Inf` if `select` is false for the corresponding state,
or 0 otherwise.
"""
function potentials_select(μ::AbstractArray; select::AbstractArray)
    # select is a keyword arg so it is not differentiated
    @assert size(μ) == size(select)
    mask = @ignore_derivatives select_mask((!).(select))
    return μ .- mask
end

energies(sequences::Sequences, model::Model) = mapreduce(hcat, model.states) do state
    energies(sequences, state)
end

energies(data::Data, model::Model) = energies(data.sequences, model)

"""
    log_likelihood_samples(model, data)

Log-likelihood of each experiment, computed as sum(R[s,t] * log(N[s,t]) over s),
for each sample 't'.
"""
function log_likelihood_samples(
    model::Model,
    data::Data;
    rare_binding::Bool = false,
    batch::MiniBatch = MiniBatch(data) # current minibatch (optional)
)
    lp = log_selectivities(model, batch; rare_binding = rare_binding)
    return log_likelihood_samples(lp, model.ζ, data; batch = batch)
end

function log_likelihood_samples(
    lp::AbstractMatrix,
    ζ::AbstractVector,
    data::Data;
    batch::MiniBatch = MiniBatch(data), # current minibatch (optional)
)
    lN = log_abundances(lp, ζ, data; batch = batch)
    X = @. ifelse(iszero(batch.counts), zero(batch.counts * lN), batch.counts * lN)
    return vec(data.lMt) / number_of_sequences(data) .+ mean_(X; dims = 1)
end

function log_likelihood(model::Model, data::Data; kwargs...)
    ls = log_likelihood_samples(model, data; kwargs...)
    return sum(ls)
end

function log_likelihood(
    lp::AbstractMatrix,
    ζ::AbstractVector,
    data::Data;
    batch::MiniBatch = MiniBatch(data)
)
    ls = log_likelihood_samples(lp, ζ, data; batch = batch)
    return sum(ls)
end

"""
    learn!(model, data; kwargs...)

Train a tree model on data.
"""
function learn!(
    model::Model,
    data::Data;
    opt = Adam(), # optimizer
    epochs = 1:100, # epoch indices
    reg = () -> false, # regularization
    batchsize = number_of_sequences(data),
    callback = nothing,
    history = MVHistory(),
    ps = params(model), # parameters to optimize over
    rare_binding::Bool = false
)
    @assert number_of_rounds(model) == number_of_rounds(data)
    _epochs = epochs2range(epochs)
    for epoch = _epochs
        for batch in minibatches(data, batchsize)
            gs = gradient(ps) do
                ll = log_likelihood(model, data; rare_binding = rare_binding, batch = batch)
                @ignore_derivatives push!(history, :loglikelihood_batch, ll)
                @ignore_derivatives push!(history, :epoch, epoch)
                return reg() - ll
            end
            update!(opt, ps, gs)
            callback === nothing || callback()
        end
        push!(
            history,
            :loglikelihood,
            log_likelihood(model, data; rare_binding = rare_binding)
        )
    end
    return history
end

"""
    rare_binding_gauge!(model)

Same as `rare_binding_gauge`, but modifies `model` in-place.
"""
function rare_binding_gauge!(model::Model)
    a = -sum(model.select .* model.μ; dims = 1) ./ sum(model.select; dims = 1)
    b = -sum(model.washed .* model.μ; dims = 1) ./ sum(model.washed; dims = 1)
    model.μ .= model.μ .+ model.select .* a .+ model.washed .* b
    model.ζ .= model.ζ .+ dropdims(a; dims = 1) .- dropdims(b; dims = 1)
    return model
end

"""
    rare_binding_gauge(model)

In the rare-binding approximation, there's a gauge invariance between μ and ζ. We fix it
by requiring sum(μ over selected states) = sum(μ over washed states) = 0, for all rounds.
"""
function rare_binding_gauge(model::Model)
    a = -sum(model.select .* model.μ; dims = 1) ./ sum(model.select; dims = 1)
    b = -sum(model.washed .* model.μ; dims = 1) ./ sum(model.washed; dims = 1)
    μ = model.μ .+ model.select .* a .+ model.washed .* b
    ζ = model.ζ .+ dropdims(a; dims = 1) .- dropdims(b; dims = 1)
    return Model(model.states, μ, ζ, model.select, model.washed)
end

"""
    rare_binding_gauge_zeta(model)

In the rare-binding approximation, there's a gauge invariance between μ and ζ. We fix it
here by setting ζ = 0 for all rounds.
"""
function rare_binding_gauge_zeta(model::Model)
    S = sum(model.select; dims=1) # number of selected states
    N = sum(model.washed; dims=1) # number of washed states
    a = -N .* model.ζ' ./ (S + N)
    b =  S .* model.ζ' ./ (S + N)
    μ = model.μ .+ model.select .* a .+ model.washed .* b
    ζ = zero(model.ζ)
    return Model(model.states, μ, ζ, model.select, model.washed)
end

"""
    rare_binding_gauge_zeta!(model)

Same as `rare_binding_gauge_zeta`, but modifies `model` in-place.
"""
function rare_binding_gauge_zeta!(model::Model)
    S = sum(model.select; dims=1) # number of selected states
    N = sum(model.washed; dims=1) # number of washed states
    a = -N .* model.ζ' ./ (S + N)
    b =  S .* model.ζ' ./ (S + N)
    model.μ .+= model.select .* a .+ model.washed .* b
    model.ζ .= 0
    return model
end

epochs2range(epochs::UnitRange) = epochs
epochs2range(epochs::Integer) = 1:epochs
