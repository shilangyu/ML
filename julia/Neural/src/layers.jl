module layers

import Base.tail


#= A Chain connects layers and propagates the outputs =#

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

#= Fully connected dense layer =#

struct Dense{W <: AbstractArray,B <: AbstractArray}
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

#= Softmax outputs the inputs mapped to a set of probabilities =#
#= 
def dsoftmax(y):		
return y * (-y + 1) =#

struct Softmax end

function (::Softmax)(input::AbstractArray)
	normalized = input .- maximum(input)
	
	exp.(normalized) / sum(normalized)
end

Base.show(io::IO, ::Softmax) = print(io, "Softmax")

parameters(::Softmax) = 0

end
