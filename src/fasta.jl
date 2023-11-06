const AAs = "ACDEFGHIKLMNPQRSTVWY" # no gap
const AAsWithGap = AAs * "*"

@assert length(AAs) == 20
@assert length(AAsWithGap) == 21

function parse_fasta_aa(file::AbstractString)
    seqs, cnts = parse_fasta_counts(file)
    aas = collect.(nt2aa.(seqs))
    prots = onehot_multiple(aas, collect(AAs))
    # retain only sequences without gaps (excludes stop codons)
    valid_seqs = Array{Bool}(reshape(prod(sum(prots; dims=1); dims=2), length(seqs)))
    return (sequences = prots[:,:,valid_seqs], counts = cnts[valid_seqs])
end

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
    return (; sequences, counts)
end

dummy_parser(q) = Returns((; sequences = zeros(Float, q, 4, 0), counts = zeros(Int, 0)))

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
