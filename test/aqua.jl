import Aqua
import Ab4Paper2023
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(Ab4Paper2023; ambiguities = false)
end
