module activations

""" 
Activation functions as 2 element named tuples.

- `:base` function accepting `x`
- `:∂y` derivative of the function accepting `y`
"""
const ActivationTuple = NamedTuple{(:base, :∂y),Tuple{T,U}} where T where U


const sigmoid = (base = x -> 1 / (1 + exp(-x)),
	∂y = y -> y * (1 - y))

const relu = (base = x -> max(0, x),
	∂y = y -> (y > 0) * 1)

const tanh = (base = Base.tanh, 
	∂y = y -> 1 - y^2)

const leaky_relu = (base = x -> x > 0 ? x : x * 1e-2,
	∂y = y -> y > 0 ? 1 : 1e-2)

	#= 
def dsigmoid(y):
		return y * (1-y)
		
def dtanh(y):
		return 1 - y**2

def drelu(y):
		return (y > 0) * 1

def dleaky_relu(y):
		return np.where(y > 0, 1, 0.01) =#

end
