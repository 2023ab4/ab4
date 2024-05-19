### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ f00c1a4f-51fc-4c0e-bbd1-b201691f8642
import Pkg, Revise; Pkg.activate(Base.current_project())

# ╔═╡ fef2bcf6-2aa7-463c-a194-3f2841a1395b
md"# Imports"

# ╔═╡ 9bcddb57-9009-4594-ba05-85b856893741
import Ab4Paper2023

# ╔═╡ c37b2ac5-8d71-423d-805d-cc747a72c6c2
import Makie, CairoMakie

# ╔═╡ a48fbc48-7b4a-46cc-b0d5-703bd713ee94
md"# Scripts"

# ╔═╡ 7d1c4fd4-f223-4b18-9f34-24591f15d126
root = Ab4Paper2023.experiment_with_targets(; colors=["black", "blue"])

# ╔═╡ 0247e6a7-bc15-448a-ba8d-d69103f596ba
data = Ab4Paper2023.Data(root)

# ╔═╡ Cell order:
# ╠═fef2bcf6-2aa7-463c-a194-3f2841a1395b
# ╠═f00c1a4f-51fc-4c0e-bbd1-b201691f8642
# ╠═9bcddb57-9009-4594-ba05-85b856893741
# ╠═c37b2ac5-8d71-423d-805d-cc747a72c6c2
# ╠═a48fbc48-7b4a-46cc-b0d5-703bd713ee94
# ╠═7d1c4fd4-f223-4b18-9f34-24591f15d126
# ╠═0247e6a7-bc15-448a-ba8d-d69103f596ba
