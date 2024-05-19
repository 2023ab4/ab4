#= For a model containing only IndepSite modes, we can store all the fields in a single
array. This can make training be twice as fast. =#

struct IndepStates{H}
    h::H   # fields: h[a,i,w]
    z::Int # number of zero modes
    function IndepStates(h::AbstractArray{<:Any,3}, z::Int = 0)
        @assert z ≥ 0
        return new{typeof(h)}(h, z)
    end
end
Flux.trainable(states::IndepStates) = (states.h,)
@functor IndepStates
number_of_states(states::IndepStates) = size(states.h, 3) + states.z

function IndepStates(z::Int, states::IndepSite...)
    h = cat([state.h for state in states]..., dims = 3)
    return IndepStates(h, z)
end
IndepStates(states::IndepSite...) = IndepStates(0, states...)

function IndepStates(states::Union{IndepSite, ZeroEnergy}...)
    Wi = findall(isa.(states, IndepSite))
    Wz = findall(isa.(states, ZeroEnergy))
    ih = IndepStates(states[Wi]...)
    return IndepStates(ih.h, length(Wz))
end

IndepModel{St<:IndepStates,Mu,Ph} = Model{St,Mu,Ph}
number_of_states(model::IndepModel) = number_of_states(model.states)

function IndepModel(
    states::IndepStates,
    μ::AbstractMatrix,
    ζ::AbstractVector,
    select::AbstractMatrix,
    washed::AbstractMatrix
)
    return Model(states, μ, ζ, select, washed)
end

function IndepModel(
    h::AbstractArray{<:Any,3},
    z::Int,
    μ::AbstractMatrix,
    ζ::AbstractVector,
    select::AbstractMatrix,
    washed::AbstractMatrix
)
    return Model(IndepStates(h, z), μ, ζ, select, washed)
end

function IndepModel(
    h::AbstractArray{<:Any,3},
    μ::AbstractMatrix,
    ζ::AbstractVector,
    select::AbstractMatrix,
    washed::AbstractMatrix
)
    return IndepModel(h, 0, μ, ζ, select, washed)
end

function IndepModel(model::Model)
    states = IndepStates(model.states...)
    Wi = findall(isa.(model.states, IndepSite))
    Wz = findall(isa.(model.states, ZeroEnergy))
    W = vcat(Wi, Wz)
    return IndepModel(
        states,
        model.μ[W,:],
        model.ζ,
        model.select[W,:],
        model.washed[W,:]
    )
end

function split_states(model::IndepModel)
    indep_states = IndepSite.(eachslice(model.states.h; dims = 3))
    zeros_states = [ZeroEnergy() for _ = 1:model.states.z]
    states = (indep_states..., zeros_states...)
    return Model(states, model.μ, model.ζ, model.select, model.washed)
end

function energies(sequences::Sequences, states::IndepStates)
    A, L, S = size(sequences)
    Ei = -reshape(sequences, A*L, :)' * reshape(states.h, A*L, :)
    E = hcat(Ei, zeros(S, states.z))
    @assert size(E) == (S, number_of_states(states))
    return E
end

function energies(sequences::Sequences, model::IndepModel)
    return energies(sequences, model.states)
end

"""
    zerosum(model)

Applies the zerosum gauge to the fields 'h'.
"""
zerosum(h::AbstractArray; dims = 1) = h .- mean(h; dims = dims)
zerosum(state::IndepSite) = IndepSite(zerosum(state.h))
zerosum(states::IndepStates) = IndepStates(zerosum(states.h))
function zerosum(model::IndepModel)
    Δh = mean(model.states.h; dims = 1)
    h = model.states.h .- Δh
    μ = model.μ[1:size(h, 3), :] .+ reshape(sum(Δh; dims=2), :, 1)
    return IndepModel(h, model.states.z, μ, model.ζ, model.select, model.washed)
end
