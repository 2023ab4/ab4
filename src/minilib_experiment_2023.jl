#= Minilib experiment with selected probed sequences sequenced in january 2023 =#
# minilb="MINILIB_3" for precedent exp

function probed_minilib_2023_dir(; minilib::String = "MINILIB_3")
    @assert minilib == "MINILIB_3"
    return joinpath(@__DIR__, "..", "data", "validation_set", "minilib3_github")
end

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

function probed_2023_proteins(file::String = joinpath(probed_minilib_2023_dir(), "probed_candidates_finals_gen_stv_cntr.fasta"))

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

# function probed_2023_proteins(file::String=file_generated_seqs)

#     # read fasta format header
#     seqs, heads = parse_fasta_header(file)

#     # creation dict_categories
#     seqs_dict=Dict{Tuple,Vector}()
#     for (i,t) in enumerate(heads)
#         if haskey(seqs_dict,t[1:4])
#             seqs_dict[t[1:4]]=vcat(seqs_dict[t[1:4]]...,seqs[i])
#         else
#             seqs_dict[t[1:4]]=[seqs[i]]
#         end
#     end

#     df = DataFrame(origin=String[], target=String[],opt=String[],filter=[],sequences=Vector{String}[])
#     for (k,v) in seqs_dict
#         push!(df,(k...,v))
#     end

#     return seqs, heads, seqs_dict, df
# end

    # probed_minilib_2023_files() = Dict(
    #     ("in",1, "") => joinpath(probed_minilib_2023_dir(), "Input_Minilib_AA.fasta"),
    #     ("o+",1, "black")  => joinpath(probed_minilib_2023_dir(), "O+_Black_Minilib_R1_AA.fasta"),
    #     ("o-",1, "black")  => joinpath(probed_minilib_2023_dir(), "O-_Black_Minilib_R1_AA.fasta"),
    #     ("in",2, "black") => joinpath(probed_minilib_2023_dir(), "Input_Black_Minilib_R2_AA.fasta"),
    #     ("o+",2, "black")  => joinpath(probed_minilib_2023_dir(), "O+_Black_Minilib_R2_AA.fasta"),
    #     ("o-",2, "black")  => joinpath(probed_minilib_2023_dir(), "O-_Black_Minilib_R2_AA.fasta"),
    #     #("in",1, "blue") => joinpath(probed_minilib_2023_dir(), "Input_Minilib_AA.fasta"),
    #     ("o+",1, "blue")  => joinpath(probed_minilib_2023_dir(), "O+_Blue_Minilib_R1_AA.fasta"),
    #     ("o-",1, "blue")  => joinpath(probed_minilib_2023_dir(), "O-_Blue_Minilib_R1_AA.fasta"),
    #     ("in",2, "blue") => joinpath(probed_minilib_2023_dir(), "Input_Blue_Minilib_R2_AA.fasta"),
    #     ("o+",2, "blue")  => joinpath(probed_minilib_2023_dir(), "O+_Blue_Minilib_R2_AA.fasta"),
    #     ("o-",2, "blue")  => joinpath(probed_minilib_2023_dir(), "O-_Blue_Minilib_R2_AA.fasta"),
    # )

    probed_minilib_2023_files(;minilib::String="MINILIB_4") = Dict(
        ("in",1, "") => joinpath(probed_minilib_2023_dir(minilib=minilib), "Input_Minilib_DNA.fasta"),
        ("o+",1, "black")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O+_Black_Minilib_R1_DNA.fasta"),
        ("o-",1, "black")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O-_Black_Minilib_R1_DNA.fasta"),
        ("in",2, "black") => joinpath(probed_minilib_2023_dir(minilib=minilib), "Input_Black_Minilib_R2_DNA.fasta"),
        ("o+",2, "black")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O+_Black_Minilib_R2_DNA.fasta"),
        ("o-",2, "black")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O-_Black_Minilib_R2_DNA.fasta"),

        ("o+",1, "blue")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O+_Blue_Minilib_R1_DNA.fasta"),
        ("o-",1, "blue")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O-_Blue_Minilib_R1_DNA.fasta"),
        ("in",2, "blue") => joinpath(probed_minilib_2023_dir(minilib=minilib), "Input_Blue_Minilib_R2_DNA.fasta"),
        ("o+",2, "blue")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O+_Blue_Minilib_R2_DNA.fasta"),
        ("o-",2, "blue")  => joinpath(probed_minilib_2023_dir(minilib=minilib), "O-_Blue_Minilib_R2_DNA.fasta"),
    )

    #= Minilib containing 9 selected sequences generated by the model for probing, plus control sequences. =#
    function minilib_2023(; as=:protein, minilib::String="MINILIB_3" )
        @assert minilib == "MINILIB_3"
        @assert as == :protein

        # @assert as in (:codons, :protein)
        # if as == :codons
        #      parser = parse_fasta_codons
        # else
        #      parser = parse_fasta_aa
        # end
        parser = parse_fasta_aa

        FILES = probed_minilib_2023_files(minilib=minilib)
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
