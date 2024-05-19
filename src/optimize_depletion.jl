"""
    optimize_depletion!(model, data; rare_binding = false)

Trains the amplification factors (the ζ parameters),
at given fixed values of the other parameters. This is a concave
optimization problem and must have a unique solution.
"""
function optimize_depletion!(
    model::Model,
    data::Data;
    rare_binding::Bool = false,
    method = NelderMead(),
    verbose::Bool = false
)
    lp = log_selectivities(model, data; rare_binding = rare_binding)
    model.ζ .= optimize_depletion(lp, model.ζ, data; method = method, verbose = verbose)
    return model
end

function optimize_depletion(
    model::Model,
    data::Data;
    rare_binding::Bool = false,
    method = NelderMead(),
    verbose::Bool = false
)
    lp = log_selectivities(model, data; rare_binding = rare_binding)
    ζ = optimize_depletion(lp, model.ζ, data; method = method, verbose = verbose)
    return ζ
end

function optimize_depletion(lp::AbstractMatrix, data::Data; kwargs...)
    ζ0 = zeros(eltype(lp), number_of_rounds(data))
    return optimize_depletion(lp, ζ0, data; kwargs...)
end

"""
    optimize_depletion(lp, ζ0, data)
"""
function optimize_depletion(
    lp::AbstractMatrix,
    ζ0::AbstractVector,
    data::Data;
    method = NelderMead(),
    verbose::Bool = false
)
    function f(ζ) # zeroth order
        lN = log_abundances(lp, ζ, data)
        X = @. ifelse(iszero(data.counts), zero(data.counts * lN), data.counts * lN)
        Lst = mean_(X; dims = 1)
        return -sum(vec(data.lMt) / number_of_sequences(data) + Lst)
    end

    function fg!(F, G, ζ) # 1st order
        lN = log_abundances(lp, ζ, data)
        if F !== nothing
            X = @. ifelse(iszero(data.counts), zero(data.counts * lN), data.counts * lN)
            Lst = mean_(X; dims = 1)
            return -sum(vec(data.lMt) / number_of_sequences(data) + Lst)
        end
        if G !== nothing
            depletion_gradient!(G, lN, data)
        end
    end

    if method isa ZerothOrderOptimizer
        sol = optimize(f, ζ0, method, Options(show_trace = verbose))
    else
        sol = optimize(only_fg!(fg!), ζ0, method, Options(show_trace = verbose))
    end
    return sol.minimizer
end

function depletion_gradient!(
    G::AbstractVector,
    lN::AbstractMatrix,
    data::Data
)
    lNmax = maximum(lN; dims = 1)
    Nsum = exp.(lNmax) .* sum(exp.(lN .- lNmax); dims = 1)
    dZ = vec(exp.(data.lRt) .* (Nsum .- 1))
    dζ = subtree_sum(dZ, data.ancestors) / number_of_sequences(data)
    round_index = 0
    for (i, a) in enumerate(data.ancestors)
        if a > 0
            G[round_index+=1] = -dζ[i]
        end
    end
    return G
end
