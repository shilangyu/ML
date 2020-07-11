module layers

import Base.tail


#= A Chain connects layers and propagates the outputs. Behaves like a singular layer. =#

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

function gradients(c::Chain, input::AbstractArray, output::AbstractArray, error::AbstractArray)
	intermediate = []
	
	local out::AbstractArray
	
	# collect input/output of each layer in feedforward
	for layer in c.layers
		out = layer(input)
		push!(intermediate, (input, out))
		input = out
	end
	
	∇ = []

	# backpropagate the errors
	for layer in Iterators.reverse(c.layers)
		curr = pop!(intermediate)
		push!(∇, gradients(layer, curr..., error))
		error = errors(layer, curr..., error)
	end

	return ∇
end

function apply!(c::Chain, ∇)
		for (layer, grad) in zip(Iterators.reverse(c.layers), ∇)
			apply!(layer, grad)
		end
end

#= Fully connected dense layer =#

mutable struct Dense{W <: AbstractArray,B <: AbstractArray}
	weights::W
	bias::B
	
	σ
	∂σ
end

function Dense(dimensions::Pair{<:Integer,<:Integer}, σ=identity, ∂σ=identity)
	in, out = dimensions
	
	Dense(rand(out, in) * 2 .- 1, rand(out) * 2 .- 1, σ, ∂σ)
end

function (d::Dense)(input::AbstractArray)
	d.σ.(d.weights * input .+ d.bias)
end

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

function errors(d::Dense, ::AbstractArray, ::AbstractArray, error::AbstractArray)
	transpose(d.weights) * error
end

function apply!(d::Dense, ∇)
	d.weights -= ∇[:weights]
	d.bias -= ∇[:bias]
end


#= Softmax outputs the inputs mapped to a set of probabilities =#

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


function gradients(s::Softmax, input::AbstractArray, output::AbstractArray, error::AbstractArray)
	()
end

# is that safe? This softmax assumes there's a dense layer beforehand, otherwise the returned error is wrong
# softmax should be an activation function instead of a layer anyways... 
# Currently it isn't because to compute its output it has to know the whole input, not a singular neuron
function errors(s::Softmax, input::AbstractArray, output::AbstractArray, error::AbstractArray)
	@. s.∂(output) * error / input
end

function apply!(s::Softmax, ∇)

end

end
