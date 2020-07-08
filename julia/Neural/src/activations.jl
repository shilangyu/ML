module activations

#= 
Activation functions as tuples of 2 elements. 
First element is the function itself while the second is its derivative with respect to y =#

const sigmoid = (x -> 1 / (1 + exp(-x)),
	y -> y * (1 - y))

const relu = (x -> max(0, x),
	y -> (y > 0) * 1)

const tanh = (Base.tanh, 
	y -> 1 - y^2)

const leaky_relu = (x -> x > 0 ? x : x * 1e-2,
	y -> y > 0 ? 1 : 1e-2)
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
