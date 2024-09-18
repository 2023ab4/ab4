### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ dd10e3a6-4d13-4ff5-8121-a113e39db422
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 9dc4fa9e-1b65-489e-ad65-f25a2a95fb98
using Ab4Paper2023: posonly, normalize_counts

# ╔═╡ 9bd6e32f-d2ee-4c59-bacc-700b23f689c3
using AbstractTrees: isroot, PreOrderDFS

# ╔═╡ 6e06b51b-ea7c-4a1d-b930-bf986dd3e1fa
using Statistics: cor

# ╔═╡ 99ef2b31-5385-499b-8b98-42388d7b2ffc
using Makie: @L_str

# ╔═╡ 184df228-6b73-11ef-3200-3dc77d85659c
md"# Imports"

# ╔═╡ 684f7b9d-da9f-4dc7-8101-f525714afc05
import Makie, CairoMakie

# ╔═╡ d930bade-a192-48ce-81fa-470823a2fabe
import Ab4Paper2023

# ╔═╡ 02c0a1b9-bd3f-4547-9b4f-375dea0c7c62
import JLD2

# ╔═╡ fa9a063f-1e47-4b5e-86fa-90629831799f
md"# Load data"

# ╔═╡ 253c83f5-e1b8-44c2-aa48-96331a3fe564
root_full = Ab4Paper2023.experiment_with_targets();

# ╔═╡ bb191b20-f236-4494-8a7f-9a76baf3f6fd
data_full = Ab4Paper2023.Data(root_full);

# ╔═╡ 66e49bd9-cbe3-4500-bb46-419e02edebc0
node_idx_full = Dict(n.label => t for (t, n) in enumerate(PreOrderDFS(root_full)));

# ╔═╡ 31149618-1c94-40e8-975b-c744fd09b457
md"# Load trained model"

# ╔═╡ 74e7053b-ef42-4a9d-9904-94616ccbd909
pwd()

# ╔═╡ b9fa9e13-f39b-4664-a6ec-07b20f3ef89f
model = JLD2.load("../scripts/20240905_train_95/data/deep_model.jld2", "model")

# ╔═╡ 47e75781-1d6b-4ffb-a90c-7de08c1744b7
history = JLD2.load("../scripts/20240905_train_95/data/deep_model.jld2", "history")

# ╔═╡ e0ed77f6-2bca-43ca-8b07-d9b7d3dbb286
state_indices = JLD2.load("../scripts/20240905_train_95/data/deep_model.jld2", "state_indices")

# ╔═╡ c9e41d91-88d5-4fda-b805-49605d1e4475
λ_reg = JLD2.load("../scripts/20240905_train_95/data/deep_model.jld2", "λ_reg")

# ╔═╡ 853cacd9-4001-4ba9-9d58-52953b04f571
md"# Plot sequence abundances by rank"

# ╔═╡ 3e58b000-45a1-4c08-87cb-2be19cf3b323
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=1200, height=400, yscale=log10)
	Makie.lines!(ax, 1:size(data_full.counts, 1), sort(dropdims(sum(data_full.counts; dims=2); dims=2); rev=true))
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ f6c7ebe2-d4fa-41ea-84d8-1ab66f943e1f
md"# Split sequences by abundances"

# ╔═╡ 36298a23-92ff-4671-8a9c-674b6148810e
sequences_total_counts = dropdims(sum(data_full.counts; dims=2); dims=2)

# ╔═╡ 40435c24-537d-4189-bfda-faec37c247dd
sequences_total_counts_ranking = sortperm(sequences_total_counts; rev=true)

# ╔═╡ cbdf0141-6df8-44e3-a751-7dd41c4dff75
top_5_percent_sequences = sequences_total_counts_ranking[1:round(Int, 0.05 * length(sequences_total_counts_ranking))]

# ╔═╡ 469e14f6-fdc6-43ad-921a-7a2445547ded
bot_5_percent_sequences = sequences_total_counts_ranking[round(Int, 0.05 * length(sequences_total_counts_ranking)) + 1:end]

# ╔═╡ f09f60cf-204c-4b58-b601-019756194df8
data_train = Ab4Paper2023.remove_sequence_counts(data_full, top_5_percent_sequences)

# ╔═╡ 659f384b-c160-4c5f-8695-50d4c7430766
data_tests = Ab4Paper2023.remove_sequence_counts(data_full, bot_5_percent_sequences)

# ╔═╡ 6f3dff73-b76e-4bc0-9633-bd100b373839
md"# Evaluate model performance"

# ╔═╡ 6a9aa533-8b28-44d2-82ff-1b32915c4790
_count_thresh = 50 # count threshold for selectivity correlations

# ╔═╡ 0a3f7ac4-f31e-4952-878d-82e01d2a98db
for (t, n) = enumerate(PreOrderDFS(root_full))
	@show n.label
end

# ╔═╡ 5de2793d-fb69-4db2-96f7-f19e6648eb38
for (t, n) = enumerate(PreOrderDFS(root_full))
	i = node_idx_full[n.label]
	p = data_tests.ancestors[i]
	p > 0 || continue
	_flag = (data_tests.counts[:,i] .≥ _count_thresh) .& (data_tests.counts[:,p] .≥ _count_thresh)
	θ = log.(data_tests.counts[:,i] ./ data_tests.counts[:,p])

	lN = Ab4Paper2023.log_abundances(model, data_tests, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "$(n.label) -- Model cor:" cor(lp[_flag], θ[_flag])
end

# ╔═╡ c4a95677-6d2f-45e6-94ee-bf478defa07a
let fig = Makie.Figure(; font="Arial")
	_sz = 150
	
	# First plot: black + blue to predict both
	i = node_idx_full["both 2 o+"]
	p = data_full.ancestors[i]
	lN = Ab4Paper2023.log_abundances(model, data_full, rare_binding=true)
	lN_train = Ab4Paper2023.log_abundances(model, data_train, rare_binding=true)
	lN_tests = Ab4Paper2023.log_abundances(model, data_tests, rare_binding=true)

	ax = Makie.Axis(fig[1,1]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_train.counts)[:,i], exp.(lN_train[:,i]))..., markersize=2, color=:purple, label="Mix 2")
	Makie.scatter!(posonly(normalize_counts(data_tests.counts)[:,i], exp.(lN_tests[:,i]))..., markersize=2, color=:red, label="Mix 2")
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-2.0, 1.1), patchlabelgap=-1, markersize=50)
	
	i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	lN = Ab4Paper2023.log_abundances(model, data_full, rare_binding=true)
	lN_train = Ab4Paper2023.log_abundances(model, data_train, rare_binding=true)
	lN_tests = Ab4Paper2023.log_abundances(model, data_tests, rare_binding=true)
	
	ax = Makie.Axis(fig[1,2]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_train.counts)[:,i], exp.(lN_train[:,i]))..., markersize=2, color=:black, label="Black 2")
	Makie.scatter!(posonly(normalize_counts(data_tests.counts)[:,i], exp.(lN_tests[:,i]))..., markersize=2, color=:red, label="Black 2")

	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-2.0, 1.1), patchlabelgap=-1, markersize=50)
	
	i = node_idx_full["blue 2 o+"]
	p = data_full.ancestors[i]
	lN = Ab4Paper2023.log_abundances(model, data_full, rare_binding=true)
	lN_train = Ab4Paper2023.log_abundances(model, data_train, rare_binding=true)
	lN_tests = Ab4Paper2023.log_abundances(model, data_tests, rare_binding=true)
	
	ax = Makie.Axis(fig[1,3]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_train.counts)[:,i], exp.(lN_train[:,i]))..., markersize=2, color=:blue, label="Blue 2")
	Makie.scatter!(posonly(normalize_counts(data_tests.counts)[:,i], exp.(lN_tests[:,i]))..., markersize=2, color=:red, label="Blue 2")
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-0.05, -0.03))
	
	Makie.resize_to_layout!(fig)
	Makie.save("../fig/fig_95.pdf", fig)
	fig
end

# ╔═╡ Cell order:
# ╠═184df228-6b73-11ef-3200-3dc77d85659c
# ╠═dd10e3a6-4d13-4ff5-8121-a113e39db422
# ╠═684f7b9d-da9f-4dc7-8101-f525714afc05
# ╠═d930bade-a192-48ce-81fa-470823a2fabe
# ╠═02c0a1b9-bd3f-4547-9b4f-375dea0c7c62
# ╠═9dc4fa9e-1b65-489e-ad65-f25a2a95fb98
# ╠═9bd6e32f-d2ee-4c59-bacc-700b23f689c3
# ╠═6e06b51b-ea7c-4a1d-b930-bf986dd3e1fa
# ╠═99ef2b31-5385-499b-8b98-42388d7b2ffc
# ╠═fa9a063f-1e47-4b5e-86fa-90629831799f
# ╠═253c83f5-e1b8-44c2-aa48-96331a3fe564
# ╠═bb191b20-f236-4494-8a7f-9a76baf3f6fd
# ╠═66e49bd9-cbe3-4500-bb46-419e02edebc0
# ╠═31149618-1c94-40e8-975b-c744fd09b457
# ╠═74e7053b-ef42-4a9d-9904-94616ccbd909
# ╠═b9fa9e13-f39b-4664-a6ec-07b20f3ef89f
# ╠═47e75781-1d6b-4ffb-a90c-7de08c1744b7
# ╠═e0ed77f6-2bca-43ca-8b07-d9b7d3dbb286
# ╠═c9e41d91-88d5-4fda-b805-49605d1e4475
# ╠═853cacd9-4001-4ba9-9d58-52953b04f571
# ╠═3e58b000-45a1-4c08-87cb-2be19cf3b323
# ╠═f6c7ebe2-d4fa-41ea-84d8-1ab66f943e1f
# ╠═36298a23-92ff-4671-8a9c-674b6148810e
# ╠═40435c24-537d-4189-bfda-faec37c247dd
# ╠═cbdf0141-6df8-44e3-a751-7dd41c4dff75
# ╠═469e14f6-fdc6-43ad-921a-7a2445547ded
# ╠═f09f60cf-204c-4b58-b601-019756194df8
# ╠═659f384b-c160-4c5f-8695-50d4c7430766
# ╠═6f3dff73-b76e-4bc0-9633-bd100b373839
# ╠═6a9aa533-8b28-44d2-82ff-1b32915c4790
# ╠═0a3f7ac4-f31e-4952-878d-82e01d2a98db
# ╠═5de2793d-fb69-4db2-96f7-f19e6648eb38
# ╠═c4a95677-6d2f-45e6-94ee-bf478defa07a
