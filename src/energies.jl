struct IndepSite{H<:AbstractMatrix}
    h::H
end
@functor IndepSite
IndepSite(A::Int, L::Int, ::Type{T} = Float64) where {T} = IndepSite(randn(T,A,L) ./ T(√L))
function energies(sequences::Sequences, state::IndepSite)
    A, L, S = size(sequences)
    @assert size(state.h) == (A, L)
    Eh = -reshape(sequences, A * L, S)' * vec(state.h)
    @assert size(Eh) == (S,)
    return Eh
end


struct Epistasis{H<:AbstractMatrix, J<:AbstractArray{<:Any,4}}
    h::H
    J::J
end
@functor Epistasis
function Epistasis(A::Int, L::Int, ::Type{T} = Float64) where {T}
    return Epistasis(randn(T, A, L), randn(T, A, L, A, L))
end
function energies(sequences::Sequences, state::Epistasis)
    A, L, _ = size(sequences)
    @assert size(state.h) == (A, L)
    @assert size(state.J) == (A, L, A, L)
    Eh = energies(sequences, IndepSite(state.h))
    EJ = -tensordot(
        reshape(sequences, A*L, :),
        reshape(state.J, A*L, A*L),
        reshape(sequences, A*L, :)
    )
    @assert size(Eh) == size(EJ) == (size(sequences, 3),)
    return Eh + EJ
end

#
# inespecific constant energy model
#
struct ConstEnergy{E<:AbstractArray{<:Real,0}}
    e::E
end
@functor ConstEnergy
ConstEnergy(x::Real) = ConstEnergy(fill(x))
ConstEnergy(::Type{T} = Float64) where {T} = ConstEnergy(fill(zero(T)))
energies(sequences::Sequences, state::ConstEnergy) = repeat(state.e, size(sequences, 3))


#
# inespecific zero energy model
#
struct ZeroEnergy{E<:AbstractArray{<:Real,0}} end
ZeroEnergy(T::Type{<:Real} = Bool) = ZeroEnergy{Array{T,0}}()
@functor ZeroEnergy
energies(sequences::Sequences, state::ZeroEnergy) = energies(sequences, ConstEnergy(state))
function ChainRulesCore.rrule(::typeof(energies), sequences::Sequences, state::ZeroEnergy)
    zero_energies_pullback(_) = (NoTangent(), NoTangent(), NoTangent())
    return energies(sequences, state), zero_energies_pullback
end
ConstEnergy(::ZeroEnergy{E}) where {E} = ConstEnergy{E}(fill(false))


#
# deep neural network energy model
#
struct DeepEnergy{T}
    m::T
end
energies(sequences::Sequences, state::DeepEnergy) = vec(state.m(sequences))
@functor DeepEnergy


#= Represents a location-independent bias =#
struct GlobBias{H<:AbstractVector}
    h::H
end
@functor GlobBias
GlobBias(A::Int, ::Type{T} = Float64) where {T} = GlobBias(randn(T, A) ./ 100)
function energies(sequences::Sequences, state::GlobBias)
    A, L, S = size(sequences)
    @assert length(state.h) == A
    return -reshape(sequences, A * L, S)' * repeat(state.h, L)
end


#
# auto-regressive energy model
# https://arxiv.org/abs/2103.03292
#
struct SimpleAR{B<:AbstractMatrix, W<:AbstractArray{<:Any,4}, M<:AbstractArray{<:Any,4}}
    biases::B
    weights::W
    mask::M
end

function SimpleAR(biases::AbstractMatrix{<:Real}, weights::AbstractArray{<:Real,4})
    A, L = size(biases)
    @assert size(weights) == (A, L, A, L)
    mask = construct_ar_mask(A, L)
    @assert size(mask) == size(weights)
    return SimpleAR(biases, weights, mask)
end
function SimpleAR(A::Int, L::Int, ::Type{T} = Float64) where {T}
    return SimpleAR(randn(T, A, L), randn(T, A, L, A, L))
end
Flux.trainable(state::SimpleAR) = (state.biases, state.weights)
@functor SimpleAR

function energies(sequences::Sequences, state::SimpleAR)
    A, L, _ = size(sequences)
    @assert size(state.biases) == (A, L)
    @assert size(state.weights) == size(state.mask) == (A, L, A, L)
    W = state.weights .* state.mask
    Wx = reshape(reshape(W, A * L, A * L) * reshape(sequences, A * L, :), A, L, :)
    return sum_(sequences .* logsoftmax(Wx .+ state.biases; dims = 1); dims = (1,2))
end

function construct_ar_mask(A::Int, L::Int, ::Type{T} = Array{Int}) where {T}
    mask = BitArray(undef, A, L, A, L)
    mask .= false
    for i in 2:L
        mask[:, i, :, 1:i-1] .= true
    end
    return T(mask)
end


#= A "logical and" of different selection states.
Represents one or more states that must be all bound together,
"in series". That is, selection entails binding all of the contained
states.
More simply put, sums the energies of the contained states. =#
struct AndEnergy{T<:Tuple}
    s::T
    AndEnergy(states...) = new{typeof(states)}(states)
end
@functor AndEnergy
energies(sequences, state::AndEnergy) = mapreduce(+, state.s) do s
    energies(sequences, s)
end
