function experiment_with_targets(;
    colors=["black", "blue", "both"], labels_only::Bool=false,
    include_beads=true # whether to include rounds with empty beads
)
    which = :deep
    as = :protein
    constfilter = true
    fullfilter = true

    @assert as ∈ (:codons, :protein)
    @assert which ∈ (:spikein, :deep)

    all_colors = ["black", "blue", "both"]

    if as == :protein
        q = 20
        parser = labels_only ? dummy_parser(q) : parse_fasta_aa
    end

    if which == :deep
        FILES = deep_blue_black_files(; constfilter, fullfilter)
    end

    root_seqs = zeros(Float, q, 4, 0)
    root_cnts = Int[]
    for c in all_colors # colors
        seqs, cnts = parser(FILES[c,1,"in"])
        if c ∉ colors
            cnts .= 0 # only include counts in `colors`
        end
        root_seqs = cat(root_seqs, seqs; dims=3)
        append!(root_cnts, cnts)
    end
    root = Experiment(root_seqs, root_cnts, "root")
    experiments = Dict("root" => root)
    for c = all_colors
        experiments[c*"1o-"] = Experiment(parser(FILES[c,1,"o-"])..., root, "$c 1 o-")
        experiments[c*"1o+"] = Experiment(parser(FILES[c,1,"o+"])..., root, "$c 1 o+")
        experiments[c*"2in"] = Experiment(parser(FILES[c,2,"in"])..., experiments[c*"1o+"], "$c 2 in")
        experiments[c*"2o-"] = Experiment(parser(FILES[c,2,"o-"])..., experiments[c*"2in"], "$c 2 o-")
        experiments[c*"2o+"] = Experiment(parser(FILES[c,2,"o+"])..., experiments[c*"2in"], "$c 2 o+")
        if c ∉ colors
            for t = ["1o-", "1o+", "2in", "2o-", "2o+"]
                experiments["$c$t"].counts .= 0 # only include counts in `colors`
            end
        end
    end
    if !include_beads
        for c = all_colors, t in ["1o-", "2o-"]
            experiments["$c$t"].counts .= 0 # don't include rounds with empty beads
        end
    end
    return root
end

function deep_blue_black_files(; constfilter=true, fullfilter=true)
    function name(i)
        if constfilter && fullfilter
            "dna_exp_constreg_full_$i.fasta"
        elseif constfilter
            "dna_exp_constreg_$i.fasta"
        else
            "dna_exp_$i.fasta"
        end
    end
    return merge(
        Dict(("black",1,σ) => joinpath(deep_blue_black_dir(; constfilter, fullfilter), name(i)) for (σ,i) in zip(["in", "o+", "o-"], 1:3)),
        Dict(("blue", 1,σ) => joinpath(deep_blue_black_dir(; constfilter, fullfilter), name(i)) for (σ,i) in zip(["in", "o+", "o-"], 4:6)),
        Dict(("both", 1,σ) => joinpath(deep_blue_black_dir(; constfilter, fullfilter), name(i)) for (σ,i) in zip(["in", "o+", "o-"], 7:9)),
        Dict(("black",2,σ) => joinpath(deep_blue_black_dir(; constfilter, fullfilter), name(i)) for (σ,i) in zip(["in", "o+", "o-"], 10:12)),
        Dict(("blue", 2,σ) => joinpath(deep_blue_black_dir(; constfilter, fullfilter), name(i)) for (σ,i) in zip(["in", "o+", "o-"], 13:15)),
        Dict(("both", 2,σ) => joinpath(deep_blue_black_dir(; constfilter, fullfilter), name(i)) for (σ,i) in zip(["in", "o+", "o-"], 16:18))
    )
end

function deep_blue_black_dir(; constfilter=true, fullfilter=true)
    # corresponds to data at DEEP_SEQ_BLUE_BLACK_CONSTREG_FULL in sibyl
    @assert constfilter && fullfilter
    return joinpath(@__DIR__, "..", "data", "training_set")
end
