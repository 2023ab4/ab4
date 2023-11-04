"""
    log_multinomial(N; dims = :)

Log of multinomial coefficients, reduced across the given dimension of `N`.
"""
function log_multinomial(N::AbstractArray; dims = :)
    return loggamma.(sum(N; dims) .+ 1) .- sum(loggamma.(N .+ 1); dims)
end

function moving_average(d::Int, v::AbstractVector)
    n = length(v)
    return [mean(v[max(1, i - d):min(n, i + d)]) for i = 1:n]
end

moving_average(d::Int) = v -> moving_average(d, v)
