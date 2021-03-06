# ML

collection of ml stuff

all projects support the `--help` flag and `-i` flag for interaction with the trained algorithm!

- [calc](#calc)
- [OrNN](#OrNN)
- [XorNN](#XorNN)
- [utils](#utils)
  - [NN](#NN)

## calc

Fully connected neural network learns to perform basic arithmetic operations (\*, /, +, -)

- `in0`: left side number
- `in1`: right side number
- `in2`: operand used
- `out0`: computed result

Example:

1. data string: `'9 - 3'`
2. normalized inputs:

```python
[9 / 9, 3 / 9, ['+', '-', '*', '/'].indexOf('-') / 3]
```

3. feedforward the inputs
4. unmap the output

## OrNN

Fully connected neural network learns to simulate the OR function

- `in0`: left bool
- `in1`: right bool
- `out0`: out bool

Example:

1. inputs `[1, 0]`
2. feedforward the inputs

## XorNN

Fully connected neural network learns to simulate the XOR function

- `in0`: left bool
- `in1`: right bool
- `out0`: out bool

Example:

1. inputs `[1, 0]`
2. feedforward the inputs

## mnist

Fully connected neural network learns to recognize handwritten digits

- `in0..783`: pixel brightness
- `out0..9`: digit guess

Example:

1. inputs `[0.5, 0, 0.3, ..., 0.32]`
2. feedforward the inputs

## utils

Utils used across subprojects

### NN

A neural network written from scratch with numpy

- `__init__`: creates a NN based on the # of inputs, hidden layers, hidden nodes, and outputs as well as activation function and learning rate
- `NN.from_config`: creates a NN based on the # of inputs, hidden layers, hidden nodes, and outputs as well as activation function and learning rate specified in the passed yaml config file
- `feedforward`: accepts inputs computes outputs
- `backpropagate`: adjusts weights based on inputs and goal outputs
- `online_train`: starts a training sessions `n` times (online method)
- `batch_train`: starts a training sessions `i` times per epoch (batch method)
- `test_accuracy`: tests the NN (accuracy)
- `test_guesses`: tests the NN (guesses)
- `serialize`: serializes the weights
- `load_brain`: load serialized weights
