import numpy as np
import yaml
import activation_functions


class NN:
	def __init__(self, inputs, hidden_nodes, outputs, hidden_layers, actifunc, dactifunc, learning_rate):
		self.input_w = np.random.rand(hidden_nodes, inputs) * 2 - 1
		self.hidden_w = np.random.rand(
			hidden_layers - 1, hidden_nodes, hidden_nodes) * 2 - 1
		self.output_w = np.random.rand(outputs, hidden_nodes) * 2 - 1

		self.input_b = np.random.rand(hidden_nodes) * 2 - 1
		self.hidden_b = np.random.rand(hidden_layers - 1, hidden_nodes) * 2 - 1
		self.output_b = np.random.rand(outputs) * 2 - 1

		self.actifunc = actifunc
		self.dactifunc = dactifunc

		self.LR = learning_rate

	@staticmethod
	def from_config(path):
		with open(path) as f:
			data = yaml.safe_load(f)
			activation_function = data['activation_function']
			del data['activation_function']
			return NN(**data, actifunc=getattr(activation_functions, activation_function), dactifunc=getattr(activation_functions, 'd'+activation_function))

	def feedforward(self, input):
		input = np.array(input)
		self.nodesave = list(range(len(self.hidden_w) + 2))

		# input
		self.nodesave[0] = self.actifunc(
			self.input_w.dot(input) + self.input_b)

		# hidden
		for i, w, b in zip(range(len(self.hidden_w)), self.hidden_w, self.hidden_b):
			self.nodesave[i+1] = self.actifunc(w.dot(self.nodesave[i]) + b)

		# output
		self.nodesave[-1] = activation_functions.softmax(
			self.output_w.dot(self.nodesave[-2]) + self.output_b)

		return self.nodesave[-1]

	def get_deltas(self, input, goal_output):
		self.feedforward(input)
		input = np.array(input)

		error = list(range(len(self.nodesave)))

		# output
		error[-1] = goal_output - self.nodesave[-1]
		output_gradient = activation_functions.dsoftmax(
			self.nodesave[-1]) * error[-1] * self.LR
		output_delta = output_gradient[:, None].dot(self.nodesave[-2][None])

		# hidden
		hidden_gradient = list(range(len(self.hidden_w)))
		hidden_delta = list(range(len(self.hidden_w)))
		for i in range(len(self.hidden_w)):
			if i == 0:
				error[-i-2] = self.output_w.T.dot(error[-i-1])
			else:
				error[-i-2] = self.hidden_w[-i].T.dot(error[-i-1])
			hidden_gradient[-i-1] = self.dactifunc(
				self.nodesave[-i-2]) * error[-i-2] * self.LR
			hidden_delta[-i-1] = hidden_gradient[-i-1][:, None].dot(
				self.nodesave[-i-3][None])

		# input
		if len(self.hidden_w) == 0:
			error[0] = self.output_w.T.dot(error[1])
		else:
			error[0] = self.hidden_w[0].T.dot(error[1])
		input_gradient = self.dactifunc(self.nodesave[0]) * error[0] * self.LR
		input_delta = input_gradient[:, None].dot(input[None])

		return (input_delta, input_gradient), (hidden_delta, hidden_gradient), (output_delta, output_gradient)

	def adjust_weights(self, deltas):
		(input_delta, input_gradient), (hidden_delta, hidden_gradient), (output_delta,
																		 output_gradient) = deltas

		# weight adjustment
		self.output_w += output_delta
		self.output_b += output_gradient

		if len(hidden_gradient) != 0:
			self.hidden_w += hidden_delta
			self.hidden_b += hidden_gradient

		self.input_w += input_delta
		self.input_b += input_gradient

	def backpropagate(self, input, goal_output):
		deltas = self.get_deltas(input, goal_output)

		# cost
		cost = np.sum((goal_output - self.nodesave[-1]) ** 2 / 2)

		# weight adjustment
		self.adjust_weights(deltas)

		return cost

	def online_train(self, inputs, goal_outputs, amount):
		choices = np.random.randint(len(inputs), size=amount)
		for i in range(amount):
			self.backpropagate(inputs[choices[i]], goal_outputs[choices[i]])

	def batch_train(self, inputs, goal_outputs, epochs, batch_size):
		choices = np.random.randint(len(inputs), size=batch_size*epochs)
		for epoch in range(epochs):
			batch_deltas = self.get_deltas(
				inputs[choices[epoch*batch_size]], goal_outputs[choices[epoch*batch_size]])
			id = np.array(batch_deltas[0][0])
			ig = np.array(batch_deltas[0][1])
			hd = np.array(batch_deltas[1][0])
			hg = np.array(batch_deltas[1][1])
			od = np.array(batch_deltas[2][0])
			og = np.array(batch_deltas[2][1])

			for batch_nr in range(batch_size - 1):
				i = epoch*batch_size + batch_nr + 1
				batch_deltas = self.get_deltas(
					inputs[choices[i]], goal_outputs[choices[i]])
				id += np.array(batch_deltas[0][0])
				ig += np.array(batch_deltas[0][1])
				hd += np.array(batch_deltas[1][0])
				hg += np.array(batch_deltas[1][1])
				od += np.array(batch_deltas[2][0])
				og += np.array(batch_deltas[2][1])

			id /= batch_size
			ig /= batch_size
			hd /= batch_size
			hg /= batch_size
			od /= batch_size
			og /= batch_size
			self.adjust_weights(((id, ig), (hd, hg), (od, og)))

	def test_accuracy(self, inputs, goal_outputs, amount):
		total_error = np.array(inputs[0]) * 0
		choices = np.random.randint(len(inputs), size=amount)
		for i in range(amount):
			total_error += abs(self.feedforward(
				inputs[choices[i]]) - goal_outputs[choices[i]])

		return np.sum((amount - total_error) / amount) / len(total_error)

	def test_guesses(self, inputs, goal_outputs, amount):
		guessed_correctly = 0
		choices = np.random.randint(len(inputs), size=amount)
		for i in range(amount):
			curr_output = self.feedforward(inputs[choices[i]])
			curr_goal = goal_outputs[choices[i]]
			guessed_correctly += int(np.argmax(curr_output)
									 == np.argmax(curr_goal))
		return guessed_correctly / amount

	def serialize(self):
		return {
			'input_w': self.input_w.tolist(),
			'hidden_w': self.hidden_w.tolist(),
			'output_w': self.output_w.tolist(),
			'input_b': self.input_b.tolist(),
			'hidden_b': self.hidden_b.tolist(),
			'output_b': self.output_b.tolist()
		}

	def load_brain(self, brain):
		self.input_w = np.array(brain['input_w'])
		self.hidden_w = np.array(brain['hidden_w'])
		self.output_w = np.array(brain['output_w'])
		self.input_b = np.array(brain['input_b'])
		self.hidden_b = np.array(brain['hidden_b'])
		self.output_b = np.array(brain['output_b'])
