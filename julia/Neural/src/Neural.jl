module Neural

import BenchmarkTools: @btime



include("layers.jl")
include("activations.jl")

import .layers
import .activations


nn = layers.Chain(
	layers.Dense(28 * 28 => 16, activations.sigmoid...),
	layers.Dense(16 => 16, activations.leaky_relu...),
	layers.Dense(16 => 10),
	layers.softmax
)
l1 = layers.Dense(28 * 28 => 16, activations.sigmoid...)
l2 = layers.Dense(16 => 16, activations.leaky_relu...)
l3 = layers.Dense(16 => 10)
l4 = layers.softmax

println(nn)

input = rand(28 * 28)

@btime nn(input)
@btime input |> l1 |> l2 |> l3 |> l4


end # module
