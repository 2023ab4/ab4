"""
    Experiment

Contains the results of one experiment.
"""
struct Experiment{Seq, C}
    sequences::Seq # sequences[a,i,s]
    counts::Vector{C} # counts[s]
    label::String # a name to identify this experiment
    children::Vector{Experiment{Seq,C}}
    parent::Experiment{Seq,C}

    # root constructor
    function Experiment(sequences::Sequences, counts::Vector, label::String = "")
        @assert size(sequences, 3) == length(counts)
        return new{typeof(sequences), eltype(counts)}(
            sequences, counts, label, Experiment[]
        )
    end

    # child constructor
    function Experiment(
        sequences::Sequences,
        counts::Vector,
        parent::Experiment,
        label::String = ""
    )
        @assert size(sequences ,3) == length(counts)
        node = new{typeof(sequences), eltype(counts)}(
            sequences, counts, label, Experiment[], parent
        )
        push!(parent.children, node)
        return node
    end
end

AbstractTrees.children(node::Experiment) = node.children
AbstractTrees.printnode(io::IO, node::Experiment) = print(io, node.label)
isroot(node::Experiment) = !isdefined(node, :parent)
isleaf(node::Experiment) = isempty(children(node))

"""
    Data(experiments...)

Convenience constructor. Only pass root experiments.
"""
function Data(roots::Experiment...)
    nodes = collect_nodes(roots...)
    n_idx = Dict{Experiment, Int}(n => i for (i, n) in enumerate(nodes))
    ancestors = [isroot(n) ? 0 : n_idx[n.parent] for n in nodes]

    sequences = unique_sequences(cat_sequences([n.sequences for n in nodes]...))
    seq_index = Dict{typeof(sequences[:,:,1]), Int}(
        seq => s for (s, seq) in enumerate(eachslice(sequences; dims = 3))
    )

    counts = zeros(Int, size(sequences, 3), length(nodes))
    for root in roots
        @assert isroot(root)
        for n in PreOrderDFS(root)
            for (s, c) in zip(eachslice(n.sequences; dims = 3), n.counts)
                counts[seq_index[s], n_idx[n]] += c
            end
        end
    end

    return Data(sequences, counts, (ancestors...,))
end

"""
    collect_nodes(experiments...)

Traverses nodes from roots to leafs and collects them.
"""
function collect_nodes(roots::Experiment...)
    for root in roots
        @assert isroot(root)
    end
    return [n for root in roots for n in PreOrderDFS(root)]
end

unique_sequences(sequences::Sequences) = unique(sequences; dims = 3)
cat_sequences(sequences::Sequences...) = cat(sequences...; dims = 3)
