module losses


"""
Loss functions as 2 element named tuples.

- `:cost` calculates the cost
- `:error` calculates the error

Each function accepts the recieved output and the expected one (respectively)
"""
const LossTuple = NamedTuple{(:cost, :error),Tuple{T,U}} where T where U


const squared_error = (cost = (ŷ, y) -> sum((ŷ .- y).^2) / 2,
	error = (ŷ, y) -> ŷ - y)

end
