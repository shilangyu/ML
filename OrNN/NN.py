import numpy as np


class NN:
    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons, hiddenLayers, actifunc, dactifunc, LR):
        self.input_w = np.random.rand(hiddenNeurons, inputNeurons) * 2 - 1
        self.hidden_w = np.random.rand(
            hiddenLayers - 1, hiddenNeurons, hiddenNeurons) * 2 - 1
        self.output_w = np.random.rand(outputNeurons, hiddenNeurons) * 2 - 1

        self.input_b = np.random.rand(hiddenNeurons) * 2 - 1
        self.hidden_b = np.random.rand(hiddenLayers - 1, hiddenNeurons) * 2 - 1
        self.output_b = np.random.rand(outputNeurons) * 2 - 1

        self.actifunc = actifunc
        self.dactifunc = dactifunc

        self.LR = LR

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
        self.nodesave[-1] = self.actifunc(
            self.output_w.dot(self.nodesave[-2]) + self.output_b)

        return self.nodesave[-1]

    def backpropagate(self, input, goal_output):
        self.feedforward(input)
        input = np.array(input)

        # cost
        cost = np.sum((goal_output - self.nodesave[-1]) ** 2 / 2)

        error = list(range(len(self.nodesave)))

        # output
        error[-1] = goal_output - self.nodesave[-1]
        output_gradient = self.dactifunc(
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

        # weight adjustment
        self.output_w += output_delta
        self.output_b += output_gradient

        if len(hidden_gradient) != 0:
            self.hidden_w += hidden_delta
            self.hidden_b += hidden_gradient

        self.input_w += input_delta
        self.input_b += input_gradient

        return cost

    def train(self, inputs, goal_outputs, amount):
        choices = np.random.randint(len(inputs), size=amount)
        for i in range(amount):
            self.backpropagate(inputs[choices[i]], goal_outputs[choices[i]])

    def test(self, inputs, goal_outputs, amount):
        total_error = np.array(inputs[0]) * 0
        choices = np.random.randint(len(inputs), size=amount)
        for i in range(amount):
            total_error += abs(self.feedforward(
                inputs[choices[i]]) - goal_outputs[choices[i]])

        return np.sum((amount - total_error) / amount) / len(total_error)

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
