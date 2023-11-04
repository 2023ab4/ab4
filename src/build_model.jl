function build_model(states::NamedTuple, roots::Experiment...)
    # let's enforce that states are fed into a specific order
    @assert keys(states) == (:black, :blue, :common, :amplification, :deplification, :wash, :beads)
    nstates = sum(map(length, states))
    nrounds = count_rounds(roots...)

    state_ranges = states_indexes(states)

    μ = zeros(nstates, nrounds)
    ξ = zeros(nrounds)
    select = falses(nstates, nrounds)
    washed = falses(nstates, nrounds)

    r = 0
    for root in roots
        @assert isroot(root)
        for n in PreOrderDFS(root)
            isroot(n) && continue
            r += 1
            c, σ = first_and_last(split(n.label))
            if σ == "in"
                select[state_ranges.amplification, r] .= true
                washed[state_ranges.deplification, r] .= true
            elseif σ ∈ ["o-", "o+"]
                select[state_ranges.beads, r] .= true
                washed[state_ranges.wash, r] .= true
                if σ == "o+"
                    @assert c ∈ ("black", "blue", "both")
                    if c == "both"
                        select[union(state_ranges.blue, state_ranges.black), r] .= true
                    elseif c == "black"
                        select[state_ranges.black, r] .= true
                    elseif c == "blue"
                        select[state_ranges.blue, r] .= true
                    end
                    select[state_ranges.common, r] .= true
                end
            end
        end
    end

    return Model(((states...)...,), μ, ξ, select, washed)
end

count_rounds(roots::Experiment...) = count_nodes(roots...) - length(roots)

function count_nodes(roots::Experiment...)
    n = 0
    for root in roots
        @assert isroot(root)
        for _ in PreOrderDFS(root)
            n += 1
        end
    end
    return n
end

"""
    states_indexes(states)

Returns indexes of the states corres
"""
function states_indexes(states::NamedTuple)
    nstates = cumsum(values(map(length, states)))
    ranges = ntuple(length(states)) do s
        (s == 1 ? 1 : nstates[s - 1] + 1):nstates[s]
    end
    return NamedTuple{keys(states)}(ranges)
end
