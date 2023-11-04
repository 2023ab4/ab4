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

"""
	tensordot(X, W, Y)

`X*W*Y`, contracting all dimensions of `W` with the corresponding first
dimensions of `X` and `Y`, and matching the remaining last dimensions of
`X` to the remaining last dimensions of `Y`.

For example, `C[b] = sum(X[i,j,b] * W[i,j,μ,ν] * Y[μ,ν,b])`.
"""
function tensordot(X::NumArray, W::NumArray, Y::NumArray)
	xsize, ysize, bsize = tensorsizes(X, W, Y)
	Xmat = reshape(X, prod(xsize), prod(bsize))
	Ymat = reshape(Y, prod(ysize), prod(bsize))
	Wmat = reshape(W, prod(xsize), prod(ysize))
	if size(Wmat, 1) ≥ size(Wmat, 2)
		Cmat = sum(Ymat .* (Wmat' * Xmat); dims = 1)
	else
		Cmat = sum(Xmat .* (Wmat * Ymat); dims = 1)
	end
	return reshape(Cmat, bsize)
end

function tensorsizes(X::NumArray, W::NumArray, Y::NumArray)
	@assert iseven(ndims(X) + ndims(Y) - ndims(W))
	bdims = div(ndims(X) + ndims(Y) - ndims(W), 2)
	@assert ndims(X) ≥ bdims && ndims(Y) ≥ bdims
	xdims = ndims(X) - bdims
	ydims = ndims(Y) - bdims
	xsize = ntuple(d -> size(X, d), xdims)
	ysize = ntuple(d -> size(Y, d), ydims)
	bsize = ntuple(d -> size(X, d + xdims), bdims)
	@assert size(W) == (xsize..., ysize...)
	@assert size(X) == (xsize..., bsize...)
	@assert size(Y) == (ysize..., bsize...)
	return xsize, ysize, bsize
end

sum_(A::AbstractArray; dims = :) = dropdims(sum(A; dims = dims); dims = dims)
