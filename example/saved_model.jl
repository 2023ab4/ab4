import Ab4Paper2023
import JLD2

### import pre-trained model
model, ws, original_file = Ab4Paper2023.generative_model_2022();

states = (
    Ab4Paper2023.DeepEnergy(model.states[1].m),
    Ab4Paper2023.DeepEnergy(model.states[2].m),
    Ab4Paper2023.DeepEnergy(model.states[3].m),
    Ab4Paper2023.ZeroEnergy(Float32),
    Ab4Paper2023.ZeroEnergy(Float32),
    Ab4Paper2023.DeepEnergy(model.states[6].m),
)
model = Ab4Paper2023.Model(states, model.μ, model.ζ, model.select, model.washed)
JLD2.jldsave("data/saved_model.jld2"; model, ws, original_file)

model, ws, original_file = Ab4Paper2023.generative_model_2022v2()
