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

function train(; λ, train_targets, include_beads::Bool, filename::AbstractString)
    @info "Loading data"
    root = experiment_with_targets(; colors=train_targets, include_beads)
    data = Data(root)

    if isfile(filename)
        @info "Saved model found, loading ..."
        model, states = JLD2.load(filename, "model", "states")
    else
        @info "Saved model NOT found. Building new model ..."
        states = (
            black = ( DeepEnergy(Chain(flatten, Dense(20 * 4 => 20, selu), Dense(20 => 5, selu), Dense(5 => 1, selu))), ),
            blue  = ( DeepEnergy(Chain(flatten, Dense(20 * 4 => 20, selu), Dense(20 => 5, selu), Dense(5 => 1, selu))), ),
            common = ( ),
            amplification = ( DeepEnergy(Chain(flatten, Dense(20 * 4 => 20, selu), Dense(20 => 5, selu), Dense(5 => 1, selu))), ),
            deplification = ( DeepEnergy(Chain(flatten, Dense(20 * 4 => 20, selu), Dense(20 => 5, selu), Dense(5 => 1, selu))), ),
            wash = ( DeepEnergy(Chain(flatten, Dense(20 * 4 => 20, selu), Dense(20 => 5, selu), Dense(5 => 1, selu))), ),
            beads = ( DeepEnergy(Chain(flatten, Dense(20 * 4 => 20, selu), Dense(20 => 5, selu), Dense(5 => 1, selu))), ),
        )
        model = build_model(states, root)
    end
    state_indices = Dict(k => i for (i, k) in enumerate(keys(Base.structdiff(states, (; common=nothing)))))

    # L2 regularization function on deep model weights
    function reg_l2()
        w2 = zero(eltype(model.states[state_indices[:black]].m[2].weight))
        for k = (:black, :blue, :amplification, :beads)
            for l = 2:length(model.states[state_indices[k]].m)
                w2 += sum(abs2, model.states[state_indices[k]].m[l].weight)
            end
        end
        return w2
    end

    local history
    for batchsize = [800, 1000, 1500, 2000, 2500, 4000]
        @info "Training (batchsize $batchsize) ..."
        history = Ab4Paper2023.learn!(
            model, data; rare_binding=true, epochs=1:200, batchsize, opt=AdaBelief(), reg=() -> λ * reg_l2()
        )
    end

    @info "Optimize amplification factors on full data ..."
    data_full = Ab4Paper2023.Data(experiment_with_targets())
    Ab4Paper2023.optimize_depletion!(model, data_full; rare_binding=true, verbose=true);

    @info "Saving ..."
    JLD2.jldsave(filename; model, states, history)
end

## First train full model
@info "Training full model first ..."
# filename="data_init_with_full/deep_$(join(current_task.train_targets, '+'))"
# with_logger(MiniLogger(; io = "$filename.log", ioerr = "$filename.err")) do
#     train(; λ=0.1, train_targets=["black", "blue", "both"], include_beads=true, filename="$filename.jld2")
# end

@info "Training branch models in parallel ..."

## The next tasks are to train the models on the branches, leaving one-out
tasks = [
    (; train_targets=["black", "blue"], include_beads=true), # train on black, blue; predict both
    (; train_targets=["black", "both"], include_beads=true), # train on black, both; predict blue
    (; train_targets=["blue", "both"], include_beads=true), # train on blue, both; predict black
    (; train_targets=["blue"], include_beads=false), # train on blue (no beads); predict beads
    (; train_targets=["both"], include_beads=true), # train on both; predict black
]

@sync for current_task = tasks
    filename = "data_init_with_full/deep_$(join(current_task.train_targets, '+'))"

    # Copy full model to use as initial condition for each branch
    cp("data_init_with_full/deep_black+blue+both.jld2", "$filename.jld2")

    Threads.@spawn with_logger(MiniLogger(; io = "$filename.log", ioerr = "$filename.err")) do
        train(; λ=0.1, current_task.train_targets, current_task.include_beads, filename="$filename.jld2")
    end
end
