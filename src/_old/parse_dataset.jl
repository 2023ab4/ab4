
DIR_ = 

##################
#TRAINING SET
##################

TRAINING_DIR = joinpath(DIR_,"data/training_set/")

function training_files()
    name(i)="dna_exp_constreg_full_$i.fasta"
  
    return merge(
        Dict(("black",1,σ) => joinpath(TRAINING_DIR, name(i)) for (σ,i) in zip(["in", "o+", "o-"], 1:3)),
        Dict(("blue", 1,σ) => joinpath(TRAINING_DIR, name(i)) for (σ,i) in zip(["in", "o+", "o-"], 4:6)),
        Dict(("both", 1,σ) => joinpath(TRAINING_DIR, name(i)) for (σ,i) in zip(["in", "o+", "o-"], 7:9)),
        Dict(("black",2,σ) => joinpath(TRAINING_DIR, name(i)) for (σ,i) in zip(["in", "o+", "o-"], 10:12)),
        Dict(("blue", 2,σ) => joinpath(TRAINING_DIR, name(i)) for (σ,i) in zip(["in", "o+", "o-"], 13:15)),
        Dict(("both", 2,σ) => joinpath(TRAINING_DIR, name(i)) for (σ,i) in zip(["in", "o+", "o-"], 16:18))
    )
end

FILES = training_files()
dummy_parser(q) = Returns((; sequences = zeros(Float, q, 4, 0), counts = zeros(Int, 0)))

function experiment_with_targets(;
    as=:codons, colors=["black", "blue", "both"],
    labels_only::Bool=false,
    include_beads=true # whether to include rounds with empty beads
)
    @assert as ∈ (:codons, :protein)
    all_colors = ["black", "blue", "both"]
    if as == :codons
        q = 64
        parser = labels_only ? dummy_parser(q) : parse_fasta_codons
    elseif as == :protein
        q = 20
        parser = labels_only ? dummy_parser(q) : parse_fasta_aa
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
    for c in all_colors
        experiments[c*"1o-"] = Experiment(parser(FILES[c,1,"o-"])..., root, "$c 1 o-")
        experiments[c*"1o+"] = Experiment(parser(FILES[c,1,"o+"])..., root, "$c 1 o+")
        experiments[c*"2in"] = Experiment(parser(FILES[c,2,"in"])..., experiments[c*"1o+"], "$c 2 in")
        experiments[c*"2o-"] = Experiment(parser(FILES[c,2,"o-"])..., experiments[c*"2in"], "$c 2 o-")
        experiments[c*"2o+"] = Experiment(parser(FILES[c,2,"o+"])..., experiments[c*"2in"], "$c 2 o+")
        if c ∉ colors
            for t in ["1o-", "1o+", "2in", "2o-", "2o+"]
                experiments["$c$t"].counts .= 0 # only include counts in `colors`
            end
        end
    end
    if !include_beads
        for c in all_colors, t in ["1o-", "2o-"]
            experiments["$c$t"].counts .= 0 # don't include rounds with empty beads
        end
    end
    return root
end



##################
#VALIDATION SET
##################


validation_dir() = joinpath(DIR_,"data/validation_set/")

function parse_fasta_header(file::AbstractString)
    headers = Tuple[]
    sequences = String[]
    for line in eachline(file)
        if startswith(line, '>') # header
            words = split(line[3:end], '_')
            t=tuple(String.(words[1:4])...,parse(Int,words[5]),parse(Float64,words[6]),parse(Float64,words[7]))
            push!(headers, t)
        else
            push!(sequences, strip(line))
        end
    end
    return (sequences = sequences, headers = headers)
end

function probed_candidates(file::String = validation_dir() * "probed_sequences.fasta")

    # read fasta format header
    #seqs, heads = Ab4App.Small_Experiment_2023.parse_fasta_header(file)
    seqs, heads = parse_fasta_header(file)

    # creation dict_categories
    seqs_dict=Dict{Tuple,Vector}()
    energies=zeros(Float64,length(seqs),2)

    for (i,t) in enumerate(heads)

        energies[i,1:2] = [t[6] t[7]]
        if haskey(seqs_dict,t[1:4])
            seqs_dict[t[1:4]]=vcat(seqs_dict[t[1:4]]...,seqs[i])
            #en_dict[t[1:4]]=vcat(en_dict[t[1:4]],e)

        else
            seqs_dict[t[1:4]]=[seqs[i]]
            #en_dict[t[1:4]]=e

        end
    end

    df = DataFrame(origin=String[], target=String[],opt=String[],filter=[],sequences=Vector{String}[])#,energies=Matrix{Float64}[])
    for (k,v) in seqs_dict
        push!(df,(k...,v))
    end

    return seqs, heads, energies, seqs_dict, df
end


    validation_files() = Dict(
        ("in",1, "") => joinpath(validation_dir(), "Input_Minilib_DNA.fasta"),
        ("o+",1, "black")  => joinpath(validation_dir(), "O+_Black_Minilib_R1_DNA.fasta"),
        ("o-",1, "black")  => joinpath(validation_dir(), "O-_Black_Minilib_R1_DNA.fasta"),
        ("in",2, "black") => joinpath(validation_dir(), "Input_Black_Minilib_R2_DNA.fasta"),
        ("o+",2, "black")  => joinpath(validation_dir(), "O+_Black_Minilib_R2_DNA.fasta"),
        ("o-",2, "black")  => joinpath(validation_dir(), "O-_Black_Minilib_R2_DNA.fasta"),

        ("o+",1, "blue")  => joinpath(validation_dir(), "O+_Blue_Minilib_R1_DNA.fasta"),
        ("o-",1, "blue")  => joinpath(validation_dir(), "O-_Blue_Minilib_R1_DNA.fasta"),
        ("in",2, "blue") => joinpath(validation_dir(), "Input_Blue_Minilib_R2_DNA.fasta"),
        ("o+",2, "blue")  => joinpath(validation_dir(), "O+_Blue_Minilib_R2_DNA.fasta"),
        ("o-",2, "blue")  => joinpath(validation_dir(), "O-_Blue_Minilib_R2_DNA.fasta"),
    )

    #= Minilib containing 9 selected sequences generated by the model for probing, plus control sequences. =#
    function validation_dataset(; as=:protein )
        @assert as in (:codons, :protein)
        if as == :codons
             parser = parse_fasta_codons
        else
             parser = parse_fasta_aa
        end

        FILES = validation_files()
        root = Experiment(parser(FILES[("in",1, "")])..., "root")
        experiments = Dict("root" => root)
        for c in ["black", "blue"], s in ["-", "+"] # 1 in -> 1 o+-
            experiments["o$s"*"_1_"*"$c"] = Experiment(parser(FILES[("o$s",1,"$c")])..., root, "o$s"*"_1_"*"$c")
        end
        for c in ["black", "blue"] #1 o+ -> 2 in
            experiments["in_2_"*"$c"] = Experiment(parser(FILES[("in",2,"$c")])..., experiments["o+_1_"*"$c"], "in_2_"*"$c")
        end
        for c in ["black", "blue"], s in ["-", "+"] #2 in -> 2 o+-
            experiments["o$s"*"_2_"*"$c"] = Experiment(parser(FILES[("o$s",2,"$c")])..., experiments["in_2_"*"$c"], "o$s"*"_2_"*"$c")
        end
        return root
    end

#################
#parser
#################

const AAs = "ACDEFGHIKLMNPQRSTVWY" # no gap
const AAsWithGap = AAs * "*"
const NTs = "ACGT" # DNA nucleotides
const CODONS = [string(c1,c2,c3) for c1 in NTs for c2 in NTs for c3 in NTs] # all possible codons

@assert length(CODONS) == 64
@assert length(AAs) == 20
@assert length(AAsWithGap) == 21

@assert issorted(collect(AAs))
@assert issorted(collect(NTs))
@assert issorted(CODONS)

# Functions to load data

"""
    parse_fasta_counts(file)

Parse a fasta file, where the headers are annotated with sequence counts.
Returns sequences as strings.
"""
function parse_fasta_counts(file::AbstractString)
    counts = Int[]
    sequences = String[]
    for line in eachline(file)
        if startswith(line, '>') # header
            words = split(line, ' ')
            push!(counts, parse(Int, words[end]))
        else
            push!(sequences, strip(line))
        end
    end
    return (sequences = sequences, counts = counts)
end

function onehot_single(seq::AbstractVector, letters::AbstractVector)
    Array{Float}(permutedims(seq) .== letters) # faster than BitArray for matrix multiplies
end

function onehot_multiple(seqs::AbstractVector, letters::AbstractVector)
    @assert all(length.(seqs) .== length(first(seqs)))
    onehot = zeros(Float, length(letters), only(unique(length.(seqs))), length(seqs))
    for (n, seq) in enumerate(seqs)
        onehot[:,:,n] .= onehot_single(seq, letters)
    end
    return onehot
end

"""
    nt2aa(nt_seq)

Translates a nucleotide DNA sequence into a protein amino-acid sequence, using the
standard genetic code.
"""
function nt2aa(seq::AbstractString)
    dna = LongDNA{4}(seq)
    prt = translate(dna)
    return string(prt)
end

# Codons that code for valid amino-acids
const CODING_CODONS = filter(s -> nt2aa(s) ≠ "*", CODONS)
const NONCODING_CODONS = setdiff(CODONS, CODING_CODONS)

@assert issorted(CODING_CODONS)
@assert issorted(NONCODING_CODONS)

"""
    codon2aa_matrix

A matrix that converts codon sequences to amino-acid sequences.
"""
const codon2aa_matrix = Array{Float}(permutedims(only.(String.(translate.(LongDNA{4}.(CODONS))))) .== collect(AAs))

"""
    codon2aa(codon_seqs)

Given a onehot encoded set of codon sequences (each letter is a codon), returns
corresponding onehot encoded protein sequence. Uses the standard genetic code.
Non-coding codons are translated to all-zero columns.
"""
function codon2aa(codon_seqs::Union{AbstractMatrix, AbstractArray{<:Any,3}})
    @assert size(codon_seqs, 1) == length(CODONS) # 64
    @assert size(codon2aa_matrix) == (length(AAs), length(CODONS))
    aa_seq_mat = codon2aa_matrix * reshape(codon_seqs, length(CODONS), :)
    return reshape(aa_seq_mat, length(AAs), size(codon_seqs)[2:end]...)
end

function splitcodons(seq::AbstractString)
    @assert length(seq) % 3 == 0
    return [seq[i-3+1:i] for i = 3:3:length(seq)]
end

aa2onehot(seq::AbstractString) = onehot_single(collect(seq), collect(AAs))
nt2onehot(seq::AbstractString) = onehot_single(collect(seq), collect(NTs))
aa2onehot(sqs::AbstractVector{<:AbstractString}) = onehot_multiple(collect.(sqs), collect(AAs))
nt2onehot(sqs::AbstractVector{<:AbstractString}) = onehot_multiple(collect.(sqs), collect(NTs))

codon2onehot(seq::AbstractString) = onehot_single(splitcodons(seq), CODONS)
codon2onehot(sqs::AbstractVector{<:AbstractString}) = onehot_multiple(splitcodons.(sqs), CODONS)

function onehot2codon(seq::AbstractArray{<:Real})
    @assert size(seq, 1) == length(CODONS)
    A = reduce(string, [CODONS[Tuple(a)[1]] for a in argmax(seq; dims=1)]; dims=2, init="")
    if ndims(seq) == 2
        return only(A)
    else
        return reshape(A, size(seq)[3:end])
    end
end

function onehot2aa(seq::AbstractArray{<:Real})
    #= Fails for gaps, which are "onehot-encoded" as all zeros. Here this will be decdoed
    as the first alphabet letter, 'A', which is wrong. =#
    @assert length(AAs) ≤ size(seq, 1) ≤ length(AAsWithGap)
    A = reduce(string, [AAsWithGap[Tuple(a)[1]] for a in argmax(seq; dims=1)]; dims=2, init="")
    if ndims(seq) == 2
        return only(A)
    else
        return reshape(A, size(seq)[3:end])
    end
end

function parse_fasta_codons(file::AbstractString)
    seqs, cnts = parse_fasta_counts(file)
    return (sequences = onehot_multiple(splitcodons.(seqs), CODONS), counts = cnts)
end

function parse_fasta_aa(file::AbstractString)
    seqs, cnts = parse_fasta_counts(file)
    aas = collect.(nt2aa.(seqs))
    prots = onehot_multiple(aas, collect(AAs))
    # retain only sequences without gaps (excludes stop codons)
    valid_seqs = Array{Bool}(reshape(prod(sum(prots; dims=1); dims=2), length(seqs)))
    return (sequences = prots[:,:,valid_seqs], counts = cnts[valid_seqs])
end
