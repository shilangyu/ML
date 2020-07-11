using BenchmarkTools

import Neural: layers, activations, losses

suite = BenchmarkGroup()

create_nn() = layers.Chain(
	layers.Dense(28^2 => 16^3, activations.sigmoid),
	layers.Dense(16^3 => 16, activations.leaky_relu),
	layers.Dense(16 => 10),
	layers.Softmax()
)
nn = create_nn()
input = rand(28^2)
output = rand(10)

dense = "$(layers.parameters(create_nn()))-parameter dense network"

suite[dense] = BenchmarkGroup()

suite[dense]["initialization"] = @benchmarkable create_nn()

suite[dense]["feedforward"] = @benchmarkable nn(input)

out = nn(input)
suite[dense]["getting gradients"] = @benchmarkable layers.gradients(nn, input, nn(input), losses.squared_error[:error](out, output))

∇ = layers.gradients(nn, input, out, losses.squared_error[:error](out, output))
suite[dense]["applying gradients"] = @benchmarkable layers.apply!(nn, ∇)

tune!(suite)
println(run(suite))
