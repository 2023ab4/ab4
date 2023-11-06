module Ab4Paper2023

import AbstractTrees
import ChainRulesCore
import Flux
import JLD2
using CSV: File
using SpecialFunctions: loggamma
using AbstractTrees: PreOrderDFS
using BioSequences: LongDNA, translate
using Statistics: mean
using DataFrames: DataFrame
using Flux: @functor, logsoftmax
using LazyArtifacts: LazyArtifacts, @artifact_str
using ChainRulesCore: @ignore_derivatives
using LogExpFunctions: logsumexp

const Float = Float32 # float type we use
const NumArray{T<:Number,N} = AbstractArray{<:T,N}

include("fasta.jl")
include("data.jl")
include("node.jl")
include("full_experiment.jl")
include("ancestors.jl")
include("util.jl")
include("validation_experiment.jl")
include("energies.jl")
include("saved_model.jl")
include("model.jl")
include("build_model.jl")

end
