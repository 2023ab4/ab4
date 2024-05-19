import Ab4Paper2023
import Dates
import Flux
import JLD2
using Random: randperm
using ValueHistories: MVHistory

# Load data
root = Ab4Paper2023.experiment_with_targets()
data = Ab4Paper2023.Data(root)

# Use constant permutation of sequences in train/test split for reproducibility
if isfile("data/seq_perm.jld2")
    seq_perm = JLD2.load("data/seq_perm.jld2", "seq_perm")
else
    seq_perm = randperm(Ab4Paper2023.number_of_sequences(data))
    JLD2.jldsave("data/seq_perm.jld2"; seq_perm)
end

# Train/test split
train_frac = 0.8
data_train = Ab4Paper2023.remove_sequence_counts(data, seq_perm[1 + round(Int, train_frac * Ab4Paper2023.number_of_sequences(data)):end])
data_tests = Ab4Paper2023.remove_sequence_counts(data, seq_perm[1:round(Int, train_frac * Ab4Paper2023.number_of_sequences(data))])

# Train indep model
states = (
    black = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 1))), ),
    blue  = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 1))), ),
    common = ( ),
    amplification = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 1))), ),
    deplification = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 1))), ),
    wash = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 1))), ),
    beads = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 1))), ),
)

model = Ab4Paper2023.build_model(states, root)
state_indices = Dict(k => i for (i, k) in enumerate(keys(Base.structdiff(states, (; common=nothing)))))

# L2 regularization function on model weights
function reg_l2()
	w2 = zero(eltype(model.states[state_indices[:black]].m[2].weight))
	for k in (:black, :blue, :amplification, :beads)
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
    global history = Ab4Paper2023.learn!(model, data_train; rare_binding=true, epochs=1:200, batchsize, opt=Flux.AdaBelief(), reg=() -> (λ_reg * reg_l2()), history)
end

@info "$(Dates.now()) Saving"
JLD2.jldsave("data/indep_model.jld2"; model, state_indices, history, seq_perm, train_frac, λ_reg)
