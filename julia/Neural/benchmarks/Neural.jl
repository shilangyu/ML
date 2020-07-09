using BenchmarkTools

import Neural: layers, activations, losses

suite = BenchmarkGroup()

create_nn() = layers.Chain(
	layers.Dense(28^2 => 16^3, activations.sigmoid...),
	layers.Dense(16^3 => 16, activations.leaky_relu...),
	layers.Dense(16 => 10),
	layers.Softmax()
)

dense = "$(layers.parameters(create_nn()))-parameter dense network"

suite[dense] = BenchmarkGroup()

suite[dense]["initialization"] = @benchmarkable create_nn()

nn = create_nn()
input = rand(28^2)
suite[dense]["feedforward"] = @benchmarkable nn(input)


tune!(suite)
println(run(suite))
