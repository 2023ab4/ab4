### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 14333804-167d-11ef-18d5-81d8bc255d2c
import Revise, Pkg; Pkg.activate(Base.current_project())

# ╔═╡ cabdf0d3-8254-4203-bb8a-339fa6a544a7
using Makie: @L_str

# ╔═╡ 3f558665-c787-4c6e-a61d-25f60d25afd3
md"# Imports"

# ╔═╡ c9cc718b-b9e6-4248-b303-3cf194aee196
import Ab4Paper2023

# ╔═╡ d56b6413-2d05-4e75-abc8-d8159a02fa11
import Makie

# ╔═╡ 02f1c907-4a4b-49de-a645-bfc7f0ef6cb1
import CairoMakie

# ╔═╡ 1ea09621-acdb-4241-982a-891768e297be
import JLD2

# ╔═╡ f803ff89-2b8e-4393-8102-b5cf86a326aa
md"# Load data and models"

# ╔═╡ 5642d2e7-5cdb-4423-bbdc-bb698970c81a
root = Ab4Paper2023.experiment_with_targets()

# ╔═╡ 7eaac8e6-dc78-4969-8d49-507524dddb21
data = Ab4Paper2023.Data(root)

# ╔═╡ 6ac81f9e-812b-4d61-980b-7d4d9dac594c
seq_perm = JLD2.load("data/seq_perm.jld2", "seq_perm")

# ╔═╡ 0b9e6263-d2aa-48a6-bd7f-b24e53e615b5
train_frac = 0.8 # fraction of data in train set

# ╔═╡ cab8acbc-20bf-41e1-b3fa-c12f121002b5
data_train = Ab4Paper2023.remove_sequence_counts(data, seq_perm[1 + round(Int, train_frac * Ab4Paper2023.number_of_sequences(data)):end])

# ╔═╡ 9b184750-3a5f-4490-b969-41e8c8b2141c
data_tests = Ab4Paper2023.remove_sequence_counts(data, seq_perm[1:round(Int, train_frac * Ab4Paper2023.number_of_sequences(data))])

# ╔═╡ 00674de3-9a6d-495c-803e-cca8bb890fca
indep_model = JLD2.load("data/indep_model.jld2", "model")

# ╔═╡ e1d7f1f1-4f74-4eb6-bf67-439801721c9a
deep_model = JLD2.load("data/deep_model.jld2", "model")

# ╔═╡ 7b150136-3860-4421-a8c2-9edc3987b723
potts_model = JLD2.load("data/potts_model.jld2", "model")

# ╔═╡ dada1dbf-fb4f-4852-8e54-3214e5a6f1f7
indep_history = JLD2.load("data/indep_model.jld2", "history")

# ╔═╡ 99b45d7b-b447-45dd-ad32-a70efb35b2a2
deep_history = JLD2.load("data/deep_model.jld2", "history")

# ╔═╡ 2bb4c8ac-d871-4060-91b8-7711148d93d6
potts_history = JLD2.load("data/potts_model.jld2", "history")

# ╔═╡ 92fcb2e2-0313-4ff0-a9e8-e47b0b9fa0a7
md"# Plots"

# ╔═╡ 968023b7-333b-4919-bcaa-a33e28c31e50
let fig = Makie.Figure()
	history = indep_history
	ax = Makie.Axis(fig[1,1], width=400, height=200, xlabel=L"iter $(\times 10^5)$", ylabel="train log-L./seq.", title="Indep model")
	Makie.lines!(get(history[:loglikelihood])[1] ./ 1e5, get(history[:loglikelihood])[2], label="train log-likelihood")
	Makie.axislegend(position=:rb)
	
	ax = Makie.Axis(fig[1,2], width=400, height=200, xlabel=L"iter $(\times 10^5)$", ylabel="train log-L./seq. (minibatch)", title="Indep. model")
	Makie.lines!(get(history[:loglikelihood_batch])[1] ./ 1e5, get(history[:loglikelihood_batch])[2], linewidth=0.1, transparency=0.1, label="minibatch log-likelihood")
	Makie.lines!(get(history[:loglikelihood_batch])[1] ./ 1e5, Ab4Paper2023.moving_average(100)(get(history[:loglikelihood_batch])[2]), label="moving average")
	Makie.axislegend(position=:rb)

	history = deep_history
	ax = Makie.Axis(fig[2,1], width=400, height=200, xlabel=L"iter $(\times 10^5)$", ylabel="train log-L./seq.", title="Deep model")
	Makie.lines!(get(history[:loglikelihood])[1] ./ 1e5, get(history[:loglikelihood])[2], label="train log-likelihood")
	Makie.axislegend(position=:rb)
	
	ax = Makie.Axis(fig[2,2], width=400, height=200, xlabel=L"iter $(\times 10^5)$", ylabel="train log-L./seq. (minibatch)", title="Deep model")
	Makie.lines!(get(history[:loglikelihood_batch])[1] ./ 1e5, get(history[:loglikelihood_batch])[2], linewidth=0.1, transparency=0.1, label="minibatch log-likelihood")
	Makie.lines!(get(history[:loglikelihood_batch])[1] ./ 1e5, Ab4Paper2023.moving_average(100)(get(history[:loglikelihood_batch])[2]), label="moving average")
	Makie.axislegend(position=:rb)

	history = potts_history
	ax = Makie.Axis(fig[3,1], width=400, height=200, xlabel=L"iter $(\times 10^5)$", ylabel="train log-L./seq.", title="Potts model")
	Makie.lines!(get(history[:loglikelihood])[1] ./ 1e5, get(history[:loglikelihood])[2], label="train log-likelihood")
	Makie.axislegend(position=:rb)
	
	ax = Makie.Axis(fig[3,2], width=400, height=200, xlabel=L"iter $(\times 10^5)$", ylabel="train log-L./seq. (minibatch)", title="Potts model")
	Makie.lines!(get(history[:loglikelihood_batch])[1] ./ 1e5, get(history[:loglikelihood_batch])[2], linewidth=0.1, transparency=0.1, label="minibatch log-likelihood")
	Makie.lines!(get(history[:loglikelihood_batch])[1] ./ 1e5, Ab4Paper2023.moving_average(100)(get(history[:loglikelihood_batch])[2]), label="moving average")
	Makie.axislegend(position=:rb)

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 4bb7f90b-5b54-4bce-854d-cbff9a51af54
indep_model_opt_tests = Ab4Paper2023.optimize_depletion!(deepcopy(indep_model), data_tests; rare_binding=true)

# ╔═╡ de173ba9-fa04-4b38-86ac-f2d9c93e2574
deep_model_opt_tests = Ab4Paper2023.optimize_depletion!(deepcopy(deep_model), data_tests; rare_binding=true)

# ╔═╡ a18ddf66-610e-4d31-ad27-7755f59c5f94
potts_model_opt_tests = Ab4Paper2023.optimize_depletion!(deepcopy(potts_model), data_tests; rare_binding=true)

# ╔═╡ 018712ae-3058-4254-93a3-b33aa3d4cf58
Ab4Paper2023.log_likelihood(indep_model, data_train; rare_binding=true), Ab4Paper2023.log_likelihood(potts_model, data_train; rare_binding=true), Ab4Paper2023.log_likelihood(deep_model, data_train; rare_binding=true)

# ╔═╡ 2c13b5fa-1236-4f5b-963f-e80ed1b4a5ba
Ab4Paper2023.log_likelihood(indep_model_opt_tests, data_tests; rare_binding=true), Ab4Paper2023.log_likelihood(potts_model_opt_tests, data_tests; rare_binding=true), Ab4Paper2023.log_likelihood(deep_model_opt_tests, data_tests; rare_binding=true)

# ╔═╡ ed95d912-7b8d-41b7-b912-79551aeef5f1
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=300, height=300, title="Train data (80% of sequences)", xticks=(1:3, ["Indep.", "Potts", "Deep"]), ylabel="log-likelihood")
	Makie.barplot!(ax, 1:3, [
		Ab4Paper2023.log_likelihood(indep_model, data_train; rare_binding=true),
		Ab4Paper2023.log_likelihood(potts_model, data_train; rare_binding=true),
		Ab4Paper2023.log_likelihood(deep_model, data_train; rare_binding=true)
	])

	ax = Makie.Axis(fig[1,2], width=300, height=300, title="Tests data (20% of sequences)", xticks=(1:3, ["Indep.", "Potts", "Deep"]), ylabel="log-likelihood")
	Makie.barplot!(ax, 1:3, [
		Ab4Paper2023.log_likelihood(indep_model_opt_tests, data_tests; rare_binding=true),
		Ab4Paper2023.log_likelihood(potts_model_opt_tests, data_tests; rare_binding=true),
		Ab4Paper2023.log_likelihood(deep_model_opt_tests, data_tests; rare_binding=true)
	])

	Makie.resize_to_layout!(fig)
	Makie.save("figures/20240520_model_comparison.pdf", fig)
	fig
end

# ╔═╡ Cell order:
# ╠═3f558665-c787-4c6e-a61d-25f60d25afd3
# ╠═14333804-167d-11ef-18d5-81d8bc255d2c
# ╠═c9cc718b-b9e6-4248-b303-3cf194aee196
# ╠═d56b6413-2d05-4e75-abc8-d8159a02fa11
# ╠═02f1c907-4a4b-49de-a645-bfc7f0ef6cb1
# ╠═1ea09621-acdb-4241-982a-891768e297be
# ╠═cabdf0d3-8254-4203-bb8a-339fa6a544a7
# ╠═f803ff89-2b8e-4393-8102-b5cf86a326aa
# ╠═5642d2e7-5cdb-4423-bbdc-bb698970c81a
# ╠═7eaac8e6-dc78-4969-8d49-507524dddb21
# ╠═6ac81f9e-812b-4d61-980b-7d4d9dac594c
# ╠═0b9e6263-d2aa-48a6-bd7f-b24e53e615b5
# ╠═cab8acbc-20bf-41e1-b3fa-c12f121002b5
# ╠═9b184750-3a5f-4490-b969-41e8c8b2141c
# ╠═00674de3-9a6d-495c-803e-cca8bb890fca
# ╠═e1d7f1f1-4f74-4eb6-bf67-439801721c9a
# ╠═7b150136-3860-4421-a8c2-9edc3987b723
# ╠═dada1dbf-fb4f-4852-8e54-3214e5a6f1f7
# ╠═99b45d7b-b447-45dd-ad32-a70efb35b2a2
# ╠═2bb4c8ac-d871-4060-91b8-7711148d93d6
# ╠═92fcb2e2-0313-4ff0-a9e8-e47b0b9fa0a7
# ╠═968023b7-333b-4919-bcaa-a33e28c31e50
# ╠═4bb7f90b-5b54-4bce-854d-cbff9a51af54
# ╠═de173ba9-fa04-4b38-86ac-f2d9c93e2574
# ╠═a18ddf66-610e-4d31-ad27-7755f59c5f94
# ╠═018712ae-3058-4254-93a3-b33aa3d4cf58
# ╠═2c13b5fa-1236-4f5b-963f-e80ed1b4a5ba
# ╠═ed95d912-7b8d-41b7-b912-79551aeef5f1
