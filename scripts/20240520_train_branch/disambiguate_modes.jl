# In some models the two targets are symmetric and need to be deambiguated manually.

import Ab4Paper2023
import Flux
import JLD2
using Ab4Paper2023: build_model
using Ab4Paper2023: Data
using Ab4Paper2023: DeepEnergy
using Ab4Paper2023: experiment_with_targets
using Ab4Paper2023: learn!
using Ab4Paper2023: log_abundances
using Ab4Paper2023: log_likelihood
using Ab4Paper2023: moving_average
using Ab4Paper2023: normalize_counts
using Ab4Paper2023: posonly
using AbstractTrees: isroot
using AbstractTrees: PreOrderDFS
using AbstractTrees: print_tree
using Flux: AdaBelief
using Flux: Chain
using Flux: Dense
using Flux: flatten
using Flux: selu
using Logging: with_logger
using Makie: @L_str
using MiniLoggers: global_logger
using MiniLoggers: MiniLogger
using Statistics: cor
using Statistics: mean


# Load data
root_full = Ab4Paper2023.experiment_with_targets()
data_full = Ab4Paper2023.Data(root_full)
node_idx_full = Dict(n.label => t for (t, n) in enumerate(PreOrderDFS(root_full)))

_thresh = 50

#_root_dir = "data_reg_bias"
#_root_dir = "data_init_with_full"
#_root_dir = "data_indep"
_root_dir = "data_init_from_indep"

#= Model trained on blue only. Blue and beads are indescernible and must be manually disambiguated. =#
let filename = "$_root_dir/deep_blue.jld2"
    model, states, history = JLD2.load(filename, "model", "states", "history")
	model_swap, states_swap = JLD2.load(filename, "model", "states")
	states_swap = ( # swap blue / beads
        states_swap.black,
	    blue = states_swap.beads,
	    states_swap.common, states_swap.amplification, states_swap.deplification, states_swap.wash,
	    beads = states_swap.blue
	)
	model_swap = Ab4Paper2023.build_model(states_swap, root_full)

	i = node_idx_full["black 2 o-"] # predict beads
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _thresh) .& (data_full.counts[:,p] .≥ _thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])

	# original model
	lN_original = log_abundances(model, data_full, rare_binding=true)
	lp_original = lN_original[:,i] - lN_original[:,p]

	# swapped model
	lN_swapped = log_abundances(model_swap, data_full, rare_binding=true)
	lp_swapped = lN_swapped[:,i] - lN_swapped[:,p]

    if cor(lp_original[_flag], θ[_flag]) > cor(lp_swapped[_flag], θ[_flag])
        JLD2.jldsave("$filename.2"; model, states, history) # just save original model
    else
        JLD2.jldsave("$filename.2"; model=model_swap, states=states_swap, history) # save swapped model
    end
end


#= Model trained on both targets. Blue and black targets are indescernible, and must be manually disambiguated =#
let filename = "$_root_dir/deep_both.jld2"
    model, states, history = JLD2.load(filename, "model", "states", "history")
	model_swap, states_swap = JLD2.load(filename, "model", "states")
	states_swap = ( # swap black / blue
        black = states_swap.blue, blue = states_swap.black,
	    states_swap.common, states_swap.amplification, states_swap.deplification, states_swap.wash, states_swap.beads
	)
	model_swap = Ab4Paper2023.build_model(states_swap, root_full)

	i = node_idx_full["black 2 o+"] # predict black
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _thresh) .& (data_full.counts[:,p] .≥ _thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])

	# original model
	lN_original = log_abundances(model, data_full, rare_binding=true)
	lp_original = lN_original[:,i] - lN_original[:,p]

	# swapped model
	lN_swapped = log_abundances(model_swap, data_full, rare_binding=true)
	lp_swapped = lN_swapped[:,i] - lN_swapped[:,p]

    if cor(lp_original[_flag], θ[_flag]) > cor(lp_swapped[_flag], θ[_flag])
        JLD2.jldsave("$filename.2"; model, states, history) # just save original model
    else
        JLD2.jldsave("$filename.2"; model=model_swap, states=states_swap, history) # save swapped model
    end
end

#= Model trained on both targets. Black and blue are indescernible and must be manually disambiguated. =#
