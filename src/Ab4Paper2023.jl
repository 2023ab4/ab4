module Ab4Paper2023

import AbstractTrees
import CairoMakie
import ChainRulesCore
import Dates
import FiniteDifferences
import Flux
import JLD2
import LazyArtifacts
import Makie
import OneHot
import Zygote
using AbstractTrees: PreOrderDFS
using BioSequences: LongDNA
using BioSequences: translate
using ChainRulesCore: @ignore_derivatives
using ChainRulesCore: NoTangent
using CSV: File
using DataFrames: DataFrame
using Flux: @functor
using Flux: Adam
using Flux: logsoftmax
using Flux: params
using Flux: update!
using LazyArtifacts: @artifact_str
using LogExpFunctions: logsumexp
using Optim: LBFGS
using Optim: NelderMead
using Optim: only_fg!
using Optim: optimize
using Optim: Options
using Optim: ZerothOrderOptimizer
using Random: randperm
using SpecialFunctions: loggamma
using Statistics: mean
using ValueHistories: MVHistory
using Zygote: gradient

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
include("optimize_depletion.jl")
include("indep_model.jl")

end
