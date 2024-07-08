### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 3ff8c26a-bafd-4125-bbcb-15c22d01c1a7
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ 87da5cb4-480b-4898-83f2-db4ce8764467
using Ab4Paper2023: log_abundances

# ╔═╡ 956256f4-60f7-4742-a5b6-eb20e66b423b
using Ab4Paper2023: normalize_counts

# ╔═╡ f61d6086-abaf-4592-9508-809f44e66b48
using Ab4Paper2023: posonly

# ╔═╡ da2c4840-13ee-45e2-96ec-27c71ddb0385
using AbstractTrees: print_tree

# ╔═╡ 6e29ee07-3f4a-46c0-87f7-96f4059c9570
using AbstractTrees: PreOrderDFS

# ╔═╡ a2af3778-e97d-47d4-8e05-c4c0af9680ec
using AbstractTrees: isroot

# ╔═╡ 481e86b8-b084-44f3-afa8-3229808e57e1
using Statistics: cor

# ╔═╡ 5f892cf8-9985-452c-9add-07dfee96ca5b
using Makie: @L_str

# ╔═╡ 277b58a7-9b62-4894-a7f6-dcb31df710d3
md"# Imports"

# ╔═╡ 36d86a7d-b999-4c3a-a2bb-0e78a3499081
import Makie

# ╔═╡ 1f8c53da-a3d9-4540-b367-ead24fca710b
import CairoMakie

# ╔═╡ 91f1ecd4-5d77-4051-a728-9598874a3415
import JLD2

# ╔═╡ 1313a6a4-77ab-4eb2-829f-43a1b8a3df49
import Ab4Paper2023

# ╔═╡ 1d12adc3-be85-4a0a-b812-1f4f5618aa05
import Flux

# ╔═╡ Cell order:
# ╠═277b58a7-9b62-4894-a7f6-dcb31df710d3
# ╠═3ff8c26a-bafd-4125-bbcb-15c22d01c1a7
# ╠═36d86a7d-b999-4c3a-a2bb-0e78a3499081
# ╠═1f8c53da-a3d9-4540-b367-ead24fca710b
# ╠═91f1ecd4-5d77-4051-a728-9598874a3415
# ╠═1313a6a4-77ab-4eb2-829f-43a1b8a3df49
# ╠═1d12adc3-be85-4a0a-b812-1f4f5618aa05
# ╠═87da5cb4-480b-4898-83f2-db4ce8764467
# ╠═956256f4-60f7-4742-a5b6-eb20e66b423b
# ╠═f61d6086-abaf-4592-9508-809f44e66b48
# ╠═da2c4840-13ee-45e2-96ec-27c71ddb0385
# ╠═6e29ee07-3f4a-46c0-87f7-96f4059c9570
# ╠═a2af3778-e97d-47d4-8e05-c4c0af9680ec
# ╠═481e86b8-b084-44f3-afa8-3229808e57e1
# ╠═5f892cf8-9985-452c-9add-07dfee96ca5b
