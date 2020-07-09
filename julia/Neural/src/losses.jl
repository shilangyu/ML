module losses

#= 
Loss functions as tuples of 2 elements. 
First element is the function itself while the second is its derivative (which is simply the output error) =#

const squared_error = ((ŷ, y) -> sum((ŷ .- y).^2) / 2,
	(ŷ, y) -> ŷ - y)

end
