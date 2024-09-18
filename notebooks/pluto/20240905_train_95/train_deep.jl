import Ab4Paper2023
import Dates
import Flux
import JLD2
using Random: randperm
using ValueHistories: MVHistory

# Load data
root = Ab4Paper2023.experiment_with_targets()
data = Ab4Paper2023.Data(root)

# Which sequences have the highest 5% of the counts ?
sequences_total_counts = dropdims(sum(data.counts; dims=2); dims=2)
sequences_total_counts_ranking = sortperm(sequences_total_counts; rev=true)

top_5_percent_sequences = sequences_total_counts_ranking[1:round(Int, 0.05 * length(sequences_total_counts_ranking))]
bot_5_percent_sequences = sequences_total_counts_ranking[round(Int, 0.05 * length(sequences_total_counts_ranking)) + 1:end]

# Train/test split
data_train = Ab4Paper2023.remove_sequence_counts(data, top_5_percent_sequences)
data_tests = Ab4Paper2023.remove_sequence_counts(data, bot_5_percent_sequences)

# Train deep model
states = (
	black = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	blue  = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	common = ( ),
	amplification = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	deplification = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	wash = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	beads = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
)

model = Ab4Paper2023.build_model(states, root)
state_indices = Dict(k => i for (i, k) in enumerate(keys(Base.structdiff(states, (; common=nothing)))))

# L2 regularization function on deep model weights
function deep_reg_l2()
	w2 = zero(eltype(model.states[state_indices[:black]].m[2].weight))
	for k = (:black, :blue, :amplification, :deplification, :beads, :wash)
		for l in 2:length(model.states[state_indices[k]].m)
			w2 += sum(abs2, model.states[state_indices[k]].m[l].weight)
		end
	end
	return w2
end

λ_reg = 0.01

history = MVHistory()
for batchsize = [200, 1000, 2000, 4000]
    @info "$(Dates.now()) Training (batchsize $batchsize) ..."
    global history = Ab4Paper2023.learn!(model, data_train; rare_binding=true, epochs=1:200, batchsize, opt=Flux.AdaBelief(), reg=() -> (λ_reg * deep_reg_l2()), history)
end

@info "$(Dates.now()) Saving"
JLD2.jldsave("data/deep_model.jld2"; model, state_indices, history, λ_reg)
