function init_deep_from_indep(indep::DeepEnergy; b0::Real = 1000, ϵ::Real = 1e-6)
    @assert length(indep.m) == 2

    deep = DeepEnergy(Flux.Chain(
        Flux.flatten,
        Flux.Dense(20 * 4 => 20, Flux.selu),
        Flux.Dense(20 => 5, Flux.selu),
        Flux.Dense(5 => 1)
    ))

    for layer = deep.m[2:end]
        layer.weight .= 0
        layer.bias .= 0
    end

    deep.m[2].weight .= indep.m[2].weight
    deep.m[2].bias .= only(indep.m[2].bias) + b0

    deep.m[3].weight .= 1 / (size(deep.m[3].weight, 2) * Flux.NNlib.selu_λ)
    deep.m[4].weight .= 1 / (size(deep.m[4].weight, 2) * Flux.NNlib.selu_λ)

    deep.m[4].bias .= -b0

    for layer = deep.m[2:end]
        layer.weight .+= ϵ .* randn.(Float32)
    end

    return deep
end
