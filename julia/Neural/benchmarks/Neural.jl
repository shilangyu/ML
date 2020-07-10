using BenchmarkTools

import Neural: layers, activations, losses

suite = BenchmarkGroup()

create_nn() = layers.Chain(
	layers.Dense(28^2 => 16^3, activations.sigmoid...),
	layers.Dense(16^3 => 16, activations.leaky_relu...),
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

suite[dense]["backpropagation"] = @benchmarkable layers.adjust!(nn, input, output, losses.squared_error...)


tune!(suite)
println(run(suite))
