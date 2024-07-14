import Ab4Paper2023
import Flux
import JLD2
using Ab4Paper2023: build_model
using Ab4Paper2023: Data
using Ab4Paper2023: DeepEnergy
using Ab4Paper2023: experiment_with_targets
using Ab4Paper2023: train_deep_from_indep
using Ab4Paper2023: learn!
using Ab4Paper2023: log_abundances
using Ab4Paper2023: log_likelihood
using Ab4Paper2023: moving_average
using Ab4Paper2023: normalize_counts
using Ab4Paper2023: posonly
using Ab4Paper2023: deep_model_weights_l2
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
using ValueHistories: MVHistory

function train(; λ, train_targets, include_beads::Bool, filename::AbstractString, filename_indep::AbstractString)
    @info "Loading data"
    root = experiment_with_targets(; colors=train_targets, include_beads)
    data = Data(root)

    if isfile(filename)
        @info "Saved model found, loading ..."
        model, states, history = JLD2.load(filename, "model", "states", "history")
    else
        @info "Saved model NOT found. Building new model (from Indep) ..."
        model_indep, states_indep = JLD2.load(filename_indep, "model", "states")
        states = (
            black = ( train_deep_from_indep(only(states_indep.black), data.sequences; λ), ),
            blue  = ( train_deep_from_indep(only(states_indep.blue), data.sequences; λ), ),
            common = ( ),
            amplification = ( train_deep_from_indep(only(states_indep.amplification), data.sequences; λ), ),
            deplification = ( train_deep_from_indep(only(states_indep.deplification), data.sequences; λ), ),
            wash = ( train_deep_from_indep(only(states_indep.wash), data.sequences; λ), ),
            beads = ( train_deep_from_indep(only(states_indep.beads), data.sequences; λ), ),
        )
        model = build_model(states, root)
        history = MVHistory()
    end
    state_indices = Dict(k => i for (i, k) in enumerate(keys(Base.structdiff(states, (; common=nothing)))))

    # L2 regularization function on deep model weights
    function reg_l2()
        w2 = zero(eltype(model.states[state_indices[:black]].m[2].weight))
        for k = (:black, :blue, :amplification, :beads)
            w2 += deep_model_weights_l2(model.states[state_indices[k]]; include_bias=false)
        end
        return w2
    end

    for batchsize = [200, 400, 800, 1000, 1500, 2000, 2500, 4000]
        @info "Training (batchsize $batchsize) ..."
        Ab4Paper2023.learn!(
            model, data; rare_binding=true, epochs=1:700, batchsize, opt=AdaBelief(), reg=() -> λ * reg_l2(), history
        )
    end

    @info "Optimize amplification factors on full data ..."
    data_full = Ab4Paper2023.Data(experiment_with_targets())
    Ab4Paper2023.optimize_depletion!(model, data_full; rare_binding=true, verbose=true);

    @info "Saving ..."
    JLD2.jldsave(filename; model, states, history)
end

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
    filename = "data_init_from_indep_v2/deep_$(join(current_task.train_targets, '+'))"
    filename_indep = "data_indep/indep_$(join(current_task.train_targets, '+'))"

    Threads.@spawn with_logger(MiniLogger(; io = "$filename.log", ioerr = "$filename.err")) do
        train(; λ=0.01, current_task.train_targets, current_task.include_beads, filename="$filename.jld2", filename_indep="$filename_indep.jld2")
    end
end
