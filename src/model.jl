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
