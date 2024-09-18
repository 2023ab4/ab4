### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ b31223d9-cb2e-4f12-9ba0-92162f2ac5f5
import Revise, Pkg; Pkg.activate(Base.current_project())

# ╔═╡ 7c178840-cf28-43da-848f-15d6b6aacddd
using Makie: @L_str

# ╔═╡ 89ee3d6e-5671-495c-b895-005124422ed1
using Statistics: mean

# ╔═╡ bcf79dc2-9318-488d-a6e6-b4adfb708858
using Statistics: cor

# ╔═╡ f4f2c398-3618-4732-8973-0c730ad02e4c
using Ab4Paper2023: posonly

# ╔═╡ 74f90271-13cc-4461-b119-a4bd71b06e19
using Ab4Paper2023: log_abundances

# ╔═╡ a4919a83-9195-4633-94d2-2103c6406615
using Ab4Paper2023: normalize_counts

# ╔═╡ 184d23aa-c398-4f88-b649-bfcd05f94f19
using AbstractTrees: print_tree

# ╔═╡ 57c971ac-4db3-445d-9a62-327ced125ae7
using AbstractTrees: PreOrderDFS

# ╔═╡ 64ee0fa9-48d1-47fc-bd4d-a80059f355d5
using AbstractTrees: isroot

# ╔═╡ 5f8b6d22-16e9-11ef-3064-dd72faab5374
md"# Imports"

# ╔═╡ f67cd147-2af2-4d34-b329-69a80a830af7
import Ab4Paper2023

# ╔═╡ c2ba625c-a8d0-43ee-bf43-192ceb1ad603
import Makie

# ╔═╡ 1a47ee95-0665-4bea-b22a-fa857d820d66
import CairoMakie

# ╔═╡ 1a2fb77e-8bd2-4768-a26d-32570f119e77
import JLD2

# ╔═╡ 94af8544-37ea-4e21-87b4-5ff2ba4ee321
import Flux

# ╔═╡ 1bed3ac7-9125-4a88-9260-569ff9dfdc4d
import AbstractTrees

# ╔═╡ 29f8079d-c08b-4b85-86d2-ecc0e8b88a14
import HypothesisTests

# ╔═╡ 5bc22420-9fed-4c18-a921-6bbab3febef0
md"# Load data"

# ╔═╡ fd454cf4-4222-4de4-b17a-966c4c5a93e1
root_full = Ab4Paper2023.experiment_with_targets();

# ╔═╡ aa5933f0-3ab9-4fb7-9322-490c484f0f5d
data_full = Ab4Paper2023.Data(root_full);

# ╔═╡ d326c0e6-85a8-41df-9589-743315698f56
node_idx_full = Dict(n.label => t for (t, n) in enumerate(PreOrderDFS(root_full)));

# ╔═╡ 53bb3666-6e29-4782-a488-1cc0193a41d7
md"# Load models"

# ╔═╡ efe7fc00-f38f-4dc4-a454-e35b89d91db4
# model trained on black and blue data (but not both)
model_black_blue, states_black_blue = JLD2.load("../data/fig2_models/indep_black+blue_reg=0.01.jld2", "model", "states");

# ╔═╡ 521420ab-6b38-4739-b519-56ad56b581f8
# model trained on both and black data (but not blue)
model_black_both, states_black_both = JLD2.load("../data/fig2_models/indep_black+both_reg=0.01.jld2", "model", "states");

# ╔═╡ 32fb0fe4-4859-4671-8b1c-0f7533a713f2
# model trained on both and blue data (but not black)
model_blue_both, states_blue_both = JLD2.load("../data/fig2_models/indep_blue+both_reg=0.01.jld2", "model", "states");

# ╔═╡ 87e9a80f-9193-4676-bac0-b1fa8a748411
# model trained on both and blue data (but not black)
model_blue_only, states_blue_only = JLD2.load("../data/fig2_models/indep_blue_reg=0.01.jld2", "model", "states");

# ╔═╡ e8bde043-3e27-40df-bd5a-9499da917042
# model trained on both, predict black and blue
model_both, states_both = JLD2.load("../data/fig2_models/indep_both_reg=0.01.jld2", "model", "states");

# ╔═╡ 415494da-8d7f-4f6f-94fc-82ce04ff1ba7
md"# Compute correlations"

# ╔═╡ c77f5438-d89f-487e-a136-f3f5bf8929a0
count_thresh = 50

# ╔═╡ 8c478d06-05fb-46d9-a35d-386f5d3522f6
let i = node_idx_full["both 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:, i] .≥ count_thresh) .& (data_full.counts[:,p] .≥ count_thresh)
	θ = log.(data_full.counts[:, i] ./ data_full.counts[:, p])
	
	# true model
	lN = log_abundances(model_black_blue, data_full, rare_binding=true)
	lp = lN[:, i] - lN[:, p]
	
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	@info HypothesisTests.CorrelationTest(lp[_flag], θ[_flag])
end

# ╔═╡ 79a85201-9f22-424f-995e-b61f3efd749a
let i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:, i] .≥ count_thresh) .& (data_full.counts[:,p] .≥ count_thresh)
	θ = log.(data_full.counts[:, i] ./ data_full.counts[:, p])
	
	# true model
	lN = log_abundances(model_blue_both, data_full, rare_binding=true)
	lp = lN[:, i] - lN[:, p]
	
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	@info HypothesisTests.CorrelationTest(lp[_flag], θ[_flag])
end

# ╔═╡ ee389698-e347-4fcf-89d5-3ca6d2712024
let i = node_idx_full["blue 2 o-"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:, i] .≥ count_thresh) .& (data_full.counts[:,p] .≥ count_thresh)
	θ = log.(data_full.counts[:, i] ./ data_full.counts[:, p])
	
	# true model
	lN = log_abundances(model_blue_only, data_full, rare_binding=true)
	lp = lN[:, i] - lN[:, p]
	
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	@info HypothesisTests.CorrelationTest(lp[_flag], θ[_flag])
end

# ╔═╡ 718173b8-d0c8-4cf3-9035-1cef5733fc85
let i = node_idx_full["black 2 o+"]
	p = data_full.ancestors[i]
	_flag = (data_full.counts[:, i] .≥ count_thresh) .& (data_full.counts[:,p] .≥ count_thresh)
	θ = log.(data_full.counts[:, i] ./ data_full.counts[:, p])
	
	# true model
	lN = log_abundances(model_both, data_full, rare_binding=true)
	lp = lN[:, i] - lN[:, p]
	
	@info "True model cor:" cor(lp[_flag], θ[_flag])
	@info HypothesisTests.CorrelationTest(lp[_flag], θ[_flag])
end

# ╔═╡ Cell order:
# ╠═5f8b6d22-16e9-11ef-3064-dd72faab5374
# ╠═b31223d9-cb2e-4f12-9ba0-92162f2ac5f5
# ╠═f67cd147-2af2-4d34-b329-69a80a830af7
# ╠═c2ba625c-a8d0-43ee-bf43-192ceb1ad603
# ╠═1a47ee95-0665-4bea-b22a-fa857d820d66
# ╠═1a2fb77e-8bd2-4768-a26d-32570f119e77
# ╠═94af8544-37ea-4e21-87b4-5ff2ba4ee321
# ╠═1bed3ac7-9125-4a88-9260-569ff9dfdc4d
# ╠═29f8079d-c08b-4b85-86d2-ecc0e8b88a14
# ╠═7c178840-cf28-43da-848f-15d6b6aacddd
# ╠═89ee3d6e-5671-495c-b895-005124422ed1
# ╠═bcf79dc2-9318-488d-a6e6-b4adfb708858
# ╠═f4f2c398-3618-4732-8973-0c730ad02e4c
# ╠═74f90271-13cc-4461-b119-a4bd71b06e19
# ╠═a4919a83-9195-4633-94d2-2103c6406615
# ╠═184d23aa-c398-4f88-b649-bfcd05f94f19
# ╠═57c971ac-4db3-445d-9a62-327ced125ae7
# ╠═64ee0fa9-48d1-47fc-bd4d-a80059f355d5
# ╠═5bc22420-9fed-4c18-a921-6bbab3febef0
# ╠═fd454cf4-4222-4de4-b17a-966c4c5a93e1
# ╠═aa5933f0-3ab9-4fb7-9322-490c484f0f5d
# ╠═d326c0e6-85a8-41df-9589-743315698f56
# ╠═53bb3666-6e29-4782-a488-1cc0193a41d7
# ╠═efe7fc00-f38f-4dc4-a454-e35b89d91db4
# ╠═521420ab-6b38-4739-b519-56ad56b581f8
# ╠═32fb0fe4-4859-4671-8b1c-0f7533a713f2
# ╠═87e9a80f-9193-4676-bac0-b1fa8a748411
# ╠═e8bde043-3e27-40df-bd5a-9499da917042
# ╠═415494da-8d7f-4f6f-94fc-82ce04ff1ba7
# ╠═c77f5438-d89f-487e-a136-f3f5bf8929a0
# ╠═8c478d06-05fb-46d9-a35d-386f5d3522f6
# ╠═79a85201-9f22-424f-995e-b61f3efd749a
# ╠═ee389698-e347-4fcf-89d5-3ca6d2712024
# ╠═718173b8-d0c8-4cf3-9035-1cef5733fc85
