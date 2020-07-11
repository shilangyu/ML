"""
Each layer:

- can be called directly

```julia
d = Dense(10 => 5)
d(rand(10))
```

- can be pretty-printed

```julia
d = Dense(10 => 5)
println(d)
```

- has a number of parameters

```julia
d = Dense(10 => 5)
parameters(d)
```

- can calculate their gradients

```julia
d = Dense(10 => 5)
i = rand(10)
error = ...
∇ = gradients(d, i, d(i), error)
```

- can calculate their errors

```julia
d = Dense(10 => 5)
i = rand(10)
error = ...
error = errors(d, i, d(i), error)
```

- can apply the gradients

```julia
d = Dense(10 => 5)
i = rand(10)
error = ...
∇ = gradients(d, i, d(i), error)
apply!(d, ∇)
```
"""
module layers

import Base.tail
import ..activations


"""
A Chain connects layers and propagates the outputs. Behaves like a singular layer.
"""
struct Chain{T <: Tuple}
	layers::T
	
	Chain(ls...) = new{typeof(ls)}(ls)
end

propagate(::Tuple{}, y) = y
propagate(ls::Tuple, y) = propagate(tail(ls), ls[1](y))

(c::Chain)(x) = propagate(c.layers, x)

function Base.show(io::IO, c::Chain)
	print(io, "Chain(")
	join(io, c.layers, " → ")
	print(io, ")")
end

parameters(c::Chain) = sum(map(parameters, c.layers))

function gradients(c::Chain, input::AbstractArray, ::AbstractArray, error::AbstractArray)
	intermediate = []
	
	# collect input/output of each layer in feedforward
	for layer in c.layers
		output = layer(input)
		push!(intermediate, (input, output))
		input = output
	end
	
	∇ = []
	
	# backpropagate the errors
	for layer in Iterators.reverse(c.layers)
		curr = pop!(intermediate)
		pushfirst!(∇, gradients(layer, curr..., error))
		error = errors(layer, curr..., error)
	end
	
	return ∇
end

function errors(c::Chain, input::AbstractArray, ::AbstractArray, error::AbstractArray)
	intermediate = []
	
	# collect input/output of each layer in feedforward
	for layer in c.layers
		output = layer(input)
		push!(intermediate, (input, output))
		input = output
	end
	
	# backpropagate the errors
	for layer in Iterators.reverse(c.layers)
		error = errors(layer, pop!(intermediate)..., error)
	end
	
	return error
end


function apply!(c::Chain, ∇)
	for (layer, grad) in Iterators.reverse(zip(c.layers, ∇))
		apply!(layer, grad)
	end
end


"""
Fully connected dense layer.
"""
mutable struct Dense{W <: AbstractArray,B <: AbstractArray}
	weights::W
	bias::B
	
	σ
	∂σ
end

function Dense(dimensions::Pair{<:Integer,<:Integer}, func::activations.ActivationTuple=(base = identity, ∂y = identity))
	in, out = dimensions
	
	Dense(rand(out, in) * 2 .- 1, rand(out) * 2 .- 1, func...)
end

(d::Dense)(input::AbstractArray) = d.σ.(d.weights * input .+ d.bias)

function Base.show(io::IO, d::Dense)
	print(io, "Dense(", size(d.weights, 2), " => ", size(d.weights, 1))
	d.σ == identity || print(io, ", ", d.σ)
	print(io, ")")
end

parameters(d::Dense) = prod(size(d.weights)) + prod(size(d.bias))

function gradients(d::Dense, input::AbstractArray, output::AbstractArray, error::AbstractArray)
	∇ = d.∂σ.(output) .* error
	return (weights = ∇ * transpose(input), bias = ∇)
end

errors(d::Dense, ::AbstractArray, ::AbstractArray, error::AbstractArray) = transpose(d.weights) * error

function apply!(d::Dense, ∇)
	d.weights -= ∇[:weights]
	d.bias -= ∇[:bias]
end


"""
Softmax outputs the inputs mapped to a set of probabilities.
"""
struct Softmax 
	∂::Function

	Softmax() = new(y -> y * (-y + 1))
end

function (::Softmax)(input::AbstractArray)
	exponated = exp.(input .- maximum(input))
	
	exponated / sum(exponated)
end

Base.show(io::IO, ::Softmax) = print(io, "Softmax")

parameters(::Softmax) = 0


gradients(::Softmax, ::AbstractArray, ::AbstractArray, ::AbstractArray) = ()

# is that safe? This softmax assumes there's a dense layer beforehand, otherwise the returned error is wrong
# softmax should be an activation function instead of a layer anyways... 
# Currently it isn't because to compute its output it has to know the whole input, not a singular neuron
errors(s::Softmax, input::AbstractArray, output::AbstractArray, error::AbstractArray) = @. s.∂(output) * error / input

apply!(::Softmax, ∇) = ()

end
