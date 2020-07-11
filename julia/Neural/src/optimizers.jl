"""
Different optimizers that act on the gradients
"""
module optimizers



"""
Gradient descent parametrized by a single learning rate
"""
struct Descent
	α::Float64
end

optimize!(d::Descent, ∇) = ∇ .*= d.α

	
end
