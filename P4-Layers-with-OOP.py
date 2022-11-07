'''
Generalizing layers with a class
'''

import numpy as np 

np.random.seed(0)

'''
X: input (fetures), 3 samples each has 4 fetures
shape: 3*4
x1: [ 1,    2,    3,   2.5]
x2: [2.0,  5.0, -1.0,  2.0]
x3: [-1.5, 2.7,  3.3, -0.8]
X: [x1,x3,x3]

'''
X = [[ 1,    2,    3,   2.5],
     [2.0,  5.0, -1.0,  2.0],
     [-1.5, 2.7,  3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #intialize the weights randmoly
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        #intialize the biases with zeros
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        #forword path
        self.output = np.dot(inputs, self.weights) + self.biases

# Layer1 with 5 neurons and 4 inputs,  (input: 4, output: 5 for each neuron)
layer1 = Layer_Dense(4,5)

# Layer2 with 2 neurons and 5 inputs (5 inputs because this is the output from the previous Layer1),
#                                      (input: 5, output: 2 for each neuron)
layer2 = Layer_Dense(5,2)


layer1.forward(X)

print(layer1.output)

layer2.forward(layer1.output)

print('-------------------------------------')
print(layer2.output)