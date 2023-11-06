"""
    generative_model_2022()

Load the model used to generate sequences for the minilib experiment of (late) 2022.
"""
function generative_model_2022v2()
    path = joinpath(@__DIR__, "..", "data", "saved_model.jld2")
    dict = JLD2.load(path)
    return (;
        model = dict["model"],
        ws = dict["ws"],
        original_file = dict["original_file"]
    )
end
