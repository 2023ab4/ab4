### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ dd10e3a6-4d13-4ff5-8121-a113e39db422
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 2f58c97c-04c0-498f-a9b4-767aca7c9917
using AbstractTrees: PreOrderDFS

# ╔═╡ 9bd6e32f-d2ee-4c59-bacc-700b23f689c3
using AbstractTrees: isroot

# ╔═╡ 184df228-6b73-11ef-3200-3dc77d85659c
md"# Imports"

# ╔═╡ 684f7b9d-da9f-4dc7-8101-f525714afc05
import Makie, CairoMakie

# ╔═╡ d930bade-a192-48ce-81fa-470823a2fabe
import Ab4Paper2023

# ╔═╡ fa9a063f-1e47-4b5e-86fa-90629831799f
md"# Load data"

# ╔═╡ 253c83f5-e1b8-44c2-aa48-96331a3fe564
root_full = Ab4Paper2023.experiment_with_targets();

# ╔═╡ bb191b20-f236-4494-8a7f-9a76baf3f6fd
data_full = Ab4Paper2023.Data(root_full);

# ╔═╡ 66e49bd9-cbe3-4500-bb46-419e02edebc0
node_idx_full = Dict(n.label => t for (t, n) in enumerate(PreOrderDFS(root_full)));

# ╔═╡ 853cacd9-4001-4ba9-9d58-52953b04f571
md"# Plot sequence abundances by rank"

# ╔═╡ 3e58b000-45a1-4c08-87cb-2be19cf3b323
let fig = Makie.Figure()
	ax = Makie.Axis(fig[1,1], width=1200, height=400, yscale=log10)
	Makie.lines!(ax, 1:size(data_full.counts, 1), sort(dropdims(sum(data_full.counts; dims=2); dims=2); rev=true))
	Makie.resize_to_layout!(fig)
	fig
end

# ╔═╡ 08e3cbd7-3505-487e-a8cc-f5280dd1b340
sortperm(dropdims(sum(data_full.counts; dims=2); dims=2); rev=true)

# ╔═╡ faf1ee79-bed6-4bde-ab86-d8de65f19f0b
data_full.counts[1,:]

# ╔═╡ 844125e9-e8a4-4a28-9f88-fa97d218e54d
data_full.counts[72010,:]

# ╔═╡ Cell order:
# ╠═184df228-6b73-11ef-3200-3dc77d85659c
# ╠═dd10e3a6-4d13-4ff5-8121-a113e39db422
# ╠═684f7b9d-da9f-4dc7-8101-f525714afc05
# ╠═d930bade-a192-48ce-81fa-470823a2fabe
# ╠═2f58c97c-04c0-498f-a9b4-767aca7c9917
# ╠═9bd6e32f-d2ee-4c59-bacc-700b23f689c3
# ╠═fa9a063f-1e47-4b5e-86fa-90629831799f
# ╠═253c83f5-e1b8-44c2-aa48-96331a3fe564
# ╠═bb191b20-f236-4494-8a7f-9a76baf3f6fd
# ╠═66e49bd9-cbe3-4500-bb46-419e02edebc0
# ╠═853cacd9-4001-4ba9-9d58-52953b04f571
# ╠═3e58b000-45a1-4c08-87cb-2be19cf3b323
# ╠═08e3cbd7-3505-487e-a8cc-f5280dd1b340
# ╠═faf1ee79-bed6-4bde-ab86-d8de65f19f0b
# ╠═844125e9-e8a4-4a28-9f88-fa97d218e54d
