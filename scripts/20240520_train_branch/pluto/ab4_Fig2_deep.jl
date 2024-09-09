### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 3ddb9179-8674-492e-b850-de8dd3eb4283
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 5e0aeb1f-e436-4029-95eb-d1818a0ecb0f
using Ab4Paper2023: log_abundances

# ╔═╡ 66e7fb4c-b89a-4072-821b-c0014f486380
using Ab4Paper2023: normalize_counts

# ╔═╡ 0de79665-c4fe-4c01-8119-13c5d2e774c8
using Ab4Paper2023: posonly

# ╔═╡ f8725bdc-c29c-4b54-86b0-9133462d222e
using AbstractTrees: print_tree

# ╔═╡ 7bd8d045-50d6-49b8-8b7e-5bba47e41bd6
using AbstractTrees: PreOrderDFS

# ╔═╡ 974d3770-1f9f-4deb-83d0-a4c931de4b44
using AbstractTrees: isroot

# ╔═╡ 0ea37649-8da1-4b5e-b1ed-e7aea6ea6cd1
using Statistics: cor

# ╔═╡ 994957aa-3d6f-4c97-96a0-16632a9e3641
using Makie: @L_str

# ╔═╡ a7e59d38-24aa-11ef-239c-e18f270246e7
md"# Imports"

# ╔═╡ b030f607-af77-486c-b504-653b9b097fc6
import Makie

# ╔═╡ 4495175e-de65-4212-9248-9a238e3e290b
import CairoMakie

# ╔═╡ a7b3261a-f065-4cfa-be6e-377b1791daf3
import JLD2

# ╔═╡ 333170a5-515e-48fb-b715-5d29dddf2fe6
import Ab4Paper2023

# ╔═╡ 9fe9a3c2-545e-4421-8bc6-afb617fd2ff9
md"# Load data"

# ╔═╡ 950f3eb4-3391-4468-a92f-4f2b8d8c6884
root_full = Ab4Paper2023.experiment_with_targets();

# ╔═╡ 71da7d62-a509-4483-b958-018610736035
data_full = Ab4Paper2023.Data(root_full);

# ╔═╡ e11ffca4-26ef-487b-9baf-fa58bbfc7f21
node_idx_full = Dict(n.label => t for (t, n) in enumerate(PreOrderDFS(root_full)));

# ╔═╡ f5d06b63-0c10-4cc5-9ccb-a3d155fbe135
md"# Load models"

# ╔═╡ f792d37e-b46e-4b4b-bef4-8fd11f453fe5
#_models_root_dir = "../data2"
#_models_root_dir = "../data_reg_bias"
#_models_root_dir = "../data_init_with_full"
#_models_root_dir = "../data_init_from_indep"
_models_root_dir = "../data_init_from_indep_v2"

# ╔═╡ f78aaf59-44c1-4e9e-a8e2-a59807862fd2
model_black_blue, states_black_blue = JLD2.load("$_models_root_dir/deep_black+blue.jld2", "model", "states");

# ╔═╡ 271399f5-9890-4541-9379-443c91751c57
model_blue_both, states_blue_both = JLD2.load("$_models_root_dir/deep_blue+both.jld2", "model", "states");

# ╔═╡ 74c8518e-e69d-407e-96d7-d7c20329960a
model_black_both, states_black_both = JLD2.load("$_models_root_dir/deep_black+both.jld2", "model", "states");

# ╔═╡ 1a43945a-b24c-4e71-b759-38d3239a7beb
model_blue, states_blue = JLD2.load("$_models_root_dir/deep_blue.jld2.2", "model", "states");

# ╔═╡ d22033df-eb5f-462d-b7ca-cbcf43dec49a
model_both, states_both = JLD2.load("$_models_root_dir/deep_both.jld2.2", "model", "states");

# ╔═╡ 39233f9d-c23b-4f29-b90b-39f94a5fc835
#model_black_blue_both, states_black_blue_both = JLD2.load("$_models_root_dir/deep_black+blue+both.jld2", "model", "states");

# ╔═╡ f910ac5d-912c-4d97-8313-4c61e7813e90
md"# Training history"

# ╔═╡ 9c564b6b-7dfb-4302-a1f5-25c1cfe2b7e3
let fig = Makie.Figure()
	width = 250
	height = 150
	
	ax = Makie.Axis(fig[1,1]; width, height, title="Black+Blue")
	Makie.lines!(ax, get(JLD2.load("$_models_root_dir/deep_black+blue.jld2", "history")[:loglikelihood])...)
	ax = Makie.Axis(fig[1,2]; width, height, title="Blue+Both")
	Makie.lines!(ax, get(JLD2.load("$_models_root_dir/deep_blue+both.jld2", "history")[:loglikelihood])...)
	ax = Makie.Axis(fig[1,3]; width, height, title="Blue")
	Makie.lines!(ax, get(JLD2.load("$_models_root_dir/deep_blue.jld2.2", "history")[:loglikelihood])...)
	ax = Makie.Axis(fig[1,4]; width, height, title="Both")
	Makie.lines!(ax, get(JLD2.load("$_models_root_dir/deep_both.jld2.2", "history")[:loglikelihood])...)
	# ax = Makie.Axis(fig[1,5]; width, height, title="Black+Blue+Both")
	# Makie.lines!(ax, get(JLD2.load("$_models_root_dir/deep_black+blue+both.jld2", "history")[:loglikelihood])...)

	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ d19a47b6-24e5-4a24-b7f1-619635170056
md"# Models with swapped targets (as control)"

# ╔═╡ cc8ee1a0-29d3-4c58-8fbc-63840651f763
begin
	model_black_blue_swap, states_black_blue_swap = JLD2.load("$_models_root_dir/deep_black+blue.jld2", "model", "states");
	states_black_blue_swap = ( # in this case, we attempt to use only blue (removing black), to predict both
	    black = states_black_blue_swap.blue, states_black_blue_swap.blue,
	    states_black_blue_swap.common, states_black_blue_swap.amplification, states_black_blue_swap.deplification, states_black_blue_swap.wash, states_black_blue_swap.beads
	)
	model_black_blue_swap = Ab4Paper2023.build_model(states_black_blue_swap, root_full);
end

# ╔═╡ e98dc7c6-4107-4fd8-98d6-7546e9f20dc9
begin
	model_blue_both_swap, states_blue_both_swap = JLD2.load("$_models_root_dir/deep_blue+both.jld2", "model", "states");
	states_blue_both_swap = ( # control for model trained on blue+both ... black <-> blue, are swapped
	    black = states_blue_both_swap.blue,
	    blue = states_blue_both_swap.black,
	    states_blue_both_swap.common, states_blue_both_swap.amplification, states_blue_both_swap.deplification, states_blue_both_swap.wash, states_blue_both_swap.beads
	)
	model_blue_both_swap = Ab4Paper2023.build_model(states_blue_both_swap, root_full);
end

# ╔═╡ 587789ad-7471-4d38-b7c6-25445dd0d9c5
begin
	# control for model trained on blue only ... beads <-> blue, are swapped
	model_blue_only_swap, states_blue_only_swap = JLD2.load("$_models_root_dir/deep_blue.jld2.2", "model", "states");
	states_blue_only_swap = ( # swap blue / beads
	    states_blue_only_swap.black,
	    blue = states_blue_only_swap.beads,
	    states_blue_only_swap.common, states_blue_only_swap.amplification, states_blue_only_swap.deplification, states_blue_only_swap.wash,
	    beads = states_blue_only_swap.blue
	)
	model_blue_only_swap = Ab4Paper2023.build_model(states_blue_only_swap, root_full);
end

# ╔═╡ 6b455a1c-3bce-43bc-b40c-091378cf8848
begin
	model_both_swap, states_both_swap = JLD2.load("$_models_root_dir/deep_both.jld2.2", "model", "states")
	states_both_swap = ( # swap black/blue
	    black = states_both_swap.blue, blue = states_both_swap.black,
	    states_both_swap.common, states_both_swap.amplification, states_both_swap.deplification, states_both_swap.wash, states_both_swap.beads
	)
	model_both_swap = Ab4Paper2023.build_model(states_both_swap, root_full);
end

# ╔═╡ b8f9b346-574d-4114-83bb-7b35811b183e
md"# Correlations"

# ╔═╡ 28c3e2f2-cdc9-418e-b533-a99f92bfcaef
_count_thresh = 50 # count threshold for selectivity correlations

# ╔═╡ 25a959c5-fedb-4a5d-a9be-8186df3a380b
let i = node_idx_full["both 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])

	# true model
	lN = log_abundances(model_black_blue, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "True model cor:" cor(lp[_flag], θ[_flag])

	# "swapped" model -- for control
	lN = log_abundances(model_black_blue_swap, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "Swapped model cor:" cor(lp[_flag], θ[_flag])
end

# ╔═╡ fc8a28e4-008b-4849-b5f1-6f9607e70679
let i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	
	# true model
	lN = log_abundances(model_blue_both, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	
	# "swapped" model -- for control
	lN = log_abundances(model_blue_both_swap, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "Swapped model cor:" cor(lp[_flag], θ[_flag])
end

# ╔═╡ cc908bb9-43f0-4ba7-8fb4-be0c0dee9670
let i = node_idx_full["black 2 o-"] # predict beads
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	
	# true model
	lN = log_abundances(model_blue, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	
	# "swapped" model -- for control
	lN = log_abundances(model_blue_only_swap, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "Swapped model cor:" cor(lp[_flag], θ[_flag])

	#= Note that there is a symmetry in this case to the two targets. =#
end

# ╔═╡ 1106a54a-b890-4ab8-ab4f-818710dab1b9
let i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	
	# true model
	lN = log_abundances(model_both, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	
	# "swapped" model -- for control
	lN = log_abundances(model_both_swap, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	@info "Swapped model cor:" cor(lp[_flag], θ[_flag])

	#= Note that there is a symmetry in this case to the two targets. =#
end

# ╔═╡ 517e2a0f-12db-4aaa-9905-99dbd94775cb
md"""# Plots
* Try initializing with independent site model.
* Check likelihoods of models initialized at full data vs random init.
"""

# ╔═╡ 222ef044-aad3-4f79-ae76-99de49117262
let fig = Makie.Figure(; font="Arial")
	_sz = 150
	
	# First plot: black + blue to predict both
	i = node_idx_full["both 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	lN = log_abundances(model_black_blue, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	ρ = round(cor(lp[_flag], θ[_flag]); digits=2)

	ax = Makie.Axis(fig[1,1]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:purple, label="Mix 2 (cor. $ρ)")
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-2.0, 1.1), patchlabelgap=-1, markersize=50)
	
	# Second plot: train on blue and both, predict black
	i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	lN = log_abundances(model_blue_both, data_full, rare_binding=true)
	lp = lN[:,i] - lN[:,p]
	ρ = round(cor(lp[_flag], θ[_flag]); digits=2)
	
	ax = Makie.Axis(fig[1,2]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:black, label="Black 2 (corr. $ρ)")
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-2.0, 1.1), patchlabelgap=-1, markersize=50)
	
	# Third plot: train on blue (no beads), predict beads
	lN = log_abundances(model_blue, data_full, rare_binding=true)
	i = node_idx_full["black 2 o-"]
	p = data_full.ancestors[i]	
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	lp = lN[:,i] - lN[:,p]
	ρ = round(cor(lp[_flag], θ[_flag]); digits=2)
	
	ax = Makie.Axis(fig[1,3]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:orange, label="Bead 2 (corr. $ρ)")
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-0.05, -0.03))
	
	# Fourth plot: train on both, predict black
	lN = log_abundances(model_both, data_full, rare_binding=true)
	i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	θ = log.(data_full.counts[:,i] ./ data_full.counts[:,p])
	lp = lN[:,i] - lN[:,p]
	ρ = round(cor(lp[_flag], θ[_flag]); digits=2)
	
	ax = Makie.Axis(fig[1,4]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:black, label="Black 2 (corr. $ρ)")
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-0.05, -0.03))
	
	Makie.resize_to_layout!(fig)
	Makie.save("../fig/fig2_deep.pdf", fig)
	fig
end

# ╔═╡ 327886b3-c8e9-48b2-b125-05ee910b5c87
md"""
# Plots using full model
"""

# ╔═╡ 90bf7947-c65a-43bb-958f-e3318270a3ad
let fig = Makie.Figure(; font="Arial")
	_sz = 150
	
	# First plot: black + blue to predict both
	i = node_idx_full["both 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	lN = log_abundances(model_black_blue_both, data_full, rare_binding=true)

	ax = Makie.Axis(fig[1,1]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:purple)
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-2.0, 1.1), patchlabelgap=-1, markersize=50)
	
	# Second plot: train on blue and both, predict black
	i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	lN = log_abundances(model_black_blue_both, data_full, rare_binding=true)
	
	ax = Makie.Axis(fig[1,2]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:black)
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-2.0, 1.1), patchlabelgap=-1, markersize=50)
	
	# Third plot: train on blue (no beads), predict beads
	lN = log_abundances(model_black_blue_both, data_full, rare_binding=true)
	i = node_idx_full["black 2 o-"]
	p = data_full.ancestors[i]	
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	
	ax = Makie.Axis(fig[1,3]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:orange)
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-0.05, -0.03))
	
	# Fourth plot: train on both, predict black
	lN = log_abundances(model_black_blue_both, data_full, rare_binding=true)
	i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:,i] .≥ _count_thresh) .& (data_full.counts[:,p] .≥ _count_thresh)
	
	ax = Makie.Axis(fig[1,4]; xscale=log10, yscale=log10, width=_sz, height=_sz, xgridvisible=false, ygridvisible=false,
		xticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		yticks=([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"1"]),
		xlabel="observed freq.", ylabel="predicted freq."
	)
	Makie.lines!(ax, [1e-10, 1], [1e-10, 1], color=:red, linewidth=1)
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], normalize_counts(data_full.counts)[:,1])..., color=:gray, markersize=4, label="Library")
	Makie.scatter!(posonly(normalize_counts(data_full.counts)[:,i], exp.(lN[:,i]))..., markersize=2, color=:black)
	Makie.xlims!(ax, 5e-6, 1)
	Makie.ylims!(ax, 5e-6, 1)
	#Makie.axislegend(ax; framevisible=false, position=(-0.05, -0.03))
	
	Makie.resize_to_layout!(fig)
	#Makie.save("fig/fig2_indep.pdf", fig)
	fig
end

# ╔═╡ Cell order:
# ╠═a7e59d38-24aa-11ef-239c-e18f270246e7
# ╠═3ddb9179-8674-492e-b850-de8dd3eb4283
# ╠═b030f607-af77-486c-b504-653b9b097fc6
# ╠═4495175e-de65-4212-9248-9a238e3e290b
# ╠═a7b3261a-f065-4cfa-be6e-377b1791daf3
# ╠═333170a5-515e-48fb-b715-5d29dddf2fe6
# ╠═5e0aeb1f-e436-4029-95eb-d1818a0ecb0f
# ╠═66e7fb4c-b89a-4072-821b-c0014f486380
# ╠═0de79665-c4fe-4c01-8119-13c5d2e774c8
# ╠═f8725bdc-c29c-4b54-86b0-9133462d222e
# ╠═7bd8d045-50d6-49b8-8b7e-5bba47e41bd6
# ╠═974d3770-1f9f-4deb-83d0-a4c931de4b44
# ╠═0ea37649-8da1-4b5e-b1ed-e7aea6ea6cd1
# ╠═994957aa-3d6f-4c97-96a0-16632a9e3641
# ╠═9fe9a3c2-545e-4421-8bc6-afb617fd2ff9
# ╠═950f3eb4-3391-4468-a92f-4f2b8d8c6884
# ╠═71da7d62-a509-4483-b958-018610736035
# ╠═e11ffca4-26ef-487b-9baf-fa58bbfc7f21
# ╠═f5d06b63-0c10-4cc5-9ccb-a3d155fbe135
# ╠═f792d37e-b46e-4b4b-bef4-8fd11f453fe5
# ╠═f78aaf59-44c1-4e9e-a8e2-a59807862fd2
# ╠═271399f5-9890-4541-9379-443c91751c57
# ╠═74c8518e-e69d-407e-96d7-d7c20329960a
# ╠═1a43945a-b24c-4e71-b759-38d3239a7beb
# ╠═d22033df-eb5f-462d-b7ca-cbcf43dec49a
# ╠═39233f9d-c23b-4f29-b90b-39f94a5fc835
# ╠═f910ac5d-912c-4d97-8313-4c61e7813e90
# ╠═9c564b6b-7dfb-4302-a1f5-25c1cfe2b7e3
# ╠═d19a47b6-24e5-4a24-b7f1-619635170056
# ╠═cc8ee1a0-29d3-4c58-8fbc-63840651f763
# ╠═e98dc7c6-4107-4fd8-98d6-7546e9f20dc9
# ╠═587789ad-7471-4d38-b7c6-25445dd0d9c5
# ╠═6b455a1c-3bce-43bc-b40c-091378cf8848
# ╠═b8f9b346-574d-4114-83bb-7b35811b183e
# ╠═28c3e2f2-cdc9-418e-b533-a99f92bfcaef
# ╠═25a959c5-fedb-4a5d-a9be-8186df3a380b
# ╠═fc8a28e4-008b-4849-b5f1-6f9607e70679
# ╠═cc908bb9-43f0-4ba7-8fb4-be0c0dee9670
# ╠═1106a54a-b890-4ab8-ab4f-818710dab1b9
# ╠═517e2a0f-12db-4aaa-9905-99dbd94775cb
# ╠═222ef044-aad3-4f79-ae76-99de49117262
# ╠═327886b3-c8e9-48b2-b125-05ee910b5c87
# ╠═90bf7947-c65a-43bb-958f-e3318270a3ad
