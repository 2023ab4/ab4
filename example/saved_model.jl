import Ab4Paper2023
import JLD2
using Ab4Paper2023: DeepEnergy

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

function jld2reconfig2(model, states)
    states2 = (
        black = ( DeepEnergy(only(states.black).m), ),
        blue  = ( DeepEnergy(only(states.blue).m), ),
        common = states_black_blue.common,
        amplification = ( DeepEnergy(only(states.amplification).m), ),
        deplification = ( DeepEnergy(only(states.deplification).m), ),
        wash = ( DeepEnergy(only(states.wash).m), ),
        beads = ( DeepEnergy(only(states.beads).m), ),
    )

    model_states = (
        Ab4Paper2023.DeepEnergy(model.states[1].m),
        Ab4Paper2023.DeepEnergy(model.states[2].m),
        Ab4Paper2023.DeepEnergy(model.states[3].m),
        Ab4Paper2023.DeepEnergy(model.states[4].m),
        Ab4Paper2023.DeepEnergy(model.states[5].m),
        Ab4Paper2023.DeepEnergy(model.states[6].m),
    )
    model2 = Ab4Paper2023.Model(model_states, model.μ, model.ζ, model.select, model.washed)
    return model2, states2
end

# model trained on black and blue data (but not both)
model, states = jld2reconfig2(JLD2.load("data/fig2_models_old/indep_black+blue_reg=0.001.jld2", "model", "states")...);
JLD2.jldsave("data/fig2_models/indep_black+blue_reg=0.001.jld2"; model, states)

# model trained on both and black data (but not blue)
model, states =jld2reconfig2(JLD2.load("data/fig2_models_old/indep_black+both_reg=0.01.jld2", "model", "states")...);
JLD2.jldsave("data/fig2_models/indep_black+both_reg=0.01.jld2"; model, states)

# model trained on both and blue data (but not black)
model, states = jld2reconfig2(JLD2.load("data/fig2_models_old/indep_blue+both_reg=0.01.jld2", "model", "states")...);
JLD2.jldsave("data/fig2_models/indep_blue+both_reg=0.01.jld2"; model, states)

# model trained on both and blue data (but not black)
model, states = jld2reconfig2(JLD2.load("data/fig2_models_old/indep_blue_reg=0.01.jld2", "model", "states")...);
JLD2.jldsave("data/fig2_models/indep_blue_reg=0.01.jld2"; model, states)

# model trained on both, predict black and blue
model, states = jld2reconfig2(JLD2.load("data/fig2_models_old/indep_both_reg=0.01.jld2", "model", "states")...);
JLD2.jldsave("data/fig2_models/indep_both_reg=0.01.jld2"; model, states)
