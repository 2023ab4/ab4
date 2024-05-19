### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ f00c1a4f-51fc-4c0e-bbd1-b201691f8642
import Revise, Pkg; Pkg.activate(Base.current_project())

# ╔═╡ 764778b6-a0ca-46ae-9a2e-51bd50fb1fd2
using Random: randperm

# ╔═╡ d099b765-cf24-43ab-af57-d8fcfe1809ca
using ValueHistories: MVHistory

# ╔═╡ fef2bcf6-2aa7-463c-a194-3f2841a1395b
md"# Imports"

# ╔═╡ 9bcddb57-9009-4594-ba05-85b856893741
import Ab4Paper2023

# ╔═╡ b7f75dae-b12c-4f0c-b456-671c7621e737
import Flux

# ╔═╡ c37b2ac5-8d71-423d-805d-cc747a72c6c2
import Makie, CairoMakie

# ╔═╡ a48fbc48-7b4a-46cc-b0d5-703bd713ee94
md"# Load data"

# ╔═╡ 7d1c4fd4-f223-4b18-9f34-24591f15d126
root = Ab4Paper2023.experiment_with_targets()

# ╔═╡ 0247e6a7-bc15-448a-ba8d-d69103f596ba
data = Ab4Paper2023.Data(root)

# ╔═╡ 1b5ce0fa-8c11-4e14-a007-175686e6dce4
seq_perm = randperm(Ab4Paper2023.number_of_sequences(data))

# ╔═╡ 9f72e313-6bbf-403f-afb9-24b6fc804f4e
train_frac = 0.8

# ╔═╡ 680b7453-be86-4524-8ce0-9756f5eef2de
data_train = Ab4Paper2023.remove_sequence_counts(data, seq_perm[1 + round(Int, train_frac * Ab4Paper2023.number_of_sequences(data)):end])

# ╔═╡ fbbe3399-081a-4e28-a423-8b5d86bceb3e
data_tests = Ab4Paper2023.remove_sequence_counts(data, seq_perm[1:round(Int, train_frac * Ab4Paper2023.number_of_sequences(data))])

# ╔═╡ cbcea06b-7dbf-45dd-8bd7-a796db34e52d
md"# Train models"

# ╔═╡ cea3618b-e54a-4bdd-8d97-37c86e46ba47
deep_states = (
	black = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	blue  = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	common = ( ),
	amplification = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	deplification = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	wash = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
	beads = ( Ab4Paper2023.DeepEnergy(Flux.Chain(Flux.flatten, Flux.Dense(20 * 4 => 20, Flux.selu), Flux.Dense(20 => 5, Flux.selu), Flux.Dense(5 => 1, Flux.selu))), ),
)

# ╔═╡ 34e34017-ae02-4c98-aba2-d34da625f197
deep_model = Ab4Paper2023.build_model(deep_states, root)

# ╔═╡ 60a5c662-d542-4a33-a9d5-623658829f6a
deep_state_indices = Dict(k => i for (i, k) in enumerate(keys(Base.structdiff(deep_states, (; common=nothing)))))

# ╔═╡ 05c1c7f8-9264-494f-8400-af594473a78d
# L2 regularization function on deep model weights
function deep_reg_l2()
	w2 = zero(eltype(deep_model.states[deep_state_indices[:black]].m[2].weight))
	for k in (:black, :blue, :amplification, :beads)
		for l in 2:length(deep_model.states[deep_state_indices[k]].m)
			w2 += sum(abs2, deep_model.states[deep_state_indices[k]].m[l].weight)
		end
	end
	return w2
end

# ╔═╡ 38c77bf3-eebe-4bd8-8e32-28f582fbaa6b
λ_reg_deep = 0.01

# ╔═╡ e1354305-d356-49ce-8660-c588b9014443
begin
	deep_history = MVHistory()
	for batchsize = [200, 1000, 2000, 4000]
		@info "Training (batchsize $batchsize) ..."
		deep_history = Ab4Paper2023.learn!(deep_model, data_train; rare_binding=true, epochs=1:200, batchsize, opt=Flux.AdaBelief(), reg=() -> (λ_reg_deep * deep_reg_l2()), history=deep_history)
	end
end

# ╔═╡ Cell order:
# ╠═fef2bcf6-2aa7-463c-a194-3f2841a1395b
# ╠═f00c1a4f-51fc-4c0e-bbd1-b201691f8642
# ╠═9bcddb57-9009-4594-ba05-85b856893741
# ╠═b7f75dae-b12c-4f0c-b456-671c7621e737
# ╠═c37b2ac5-8d71-423d-805d-cc747a72c6c2
# ╠═764778b6-a0ca-46ae-9a2e-51bd50fb1fd2
# ╠═d099b765-cf24-43ab-af57-d8fcfe1809ca
# ╠═a48fbc48-7b4a-46cc-b0d5-703bd713ee94
# ╠═7d1c4fd4-f223-4b18-9f34-24591f15d126
# ╠═0247e6a7-bc15-448a-ba8d-d69103f596ba
# ╠═1b5ce0fa-8c11-4e14-a007-175686e6dce4
# ╠═9f72e313-6bbf-403f-afb9-24b6fc804f4e
# ╠═680b7453-be86-4524-8ce0-9756f5eef2de
# ╠═fbbe3399-081a-4e28-a423-8b5d86bceb3e
# ╠═cbcea06b-7dbf-45dd-8bd7-a796db34e52d
# ╠═cea3618b-e54a-4bdd-8d97-37c86e46ba47
# ╠═34e34017-ae02-4c98-aba2-d34da625f197
# ╠═60a5c662-d542-4a33-a9d5-623658829f6a
# ╠═05c1c7f8-9264-494f-8400-af594473a78d
# ╠═38c77bf3-eebe-4bd8-8e32-28f582fbaa6b
# ╠═e1354305-d356-49ce-8660-c588b9014443
