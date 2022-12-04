'''
Generalizing layers with a class
'''

import numpy as np 

np.random.seed(0)

'''
X: input, 4 samples each has 3 fetures
shape: 3x4
x1: [ 1,    2,    -1.5].T
x2: [2.0,  5.0,    2.7].T
x3: [3,    -1,     3.3].T
x4: [2.5,   2,    -0.8].T
X: [x1,x3,x3,x4]

'''

#shape: 3x4
X = [[ 1,    2,    3,   2.5],
     [2.0,  5.0, -1.0,  2.0],
     [-1.5, 2.7,  3.3, -0.8]]


class Layer_Dense:
    def __init__(self,  n_neurons, n_inputs):
        #intialize the weights randmoly
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)

        #intialize the biases with zeros
        self.biases = np.zeros((n_neurons, 1))


    def forward(self, inputs):
        #forword path
        self.output = np.dot(self.weights, inputs) + self.biases


# Layer1 with 5 neurons and 3 inputs features
layer1 = Layer_Dense(5,3)

# Layer2 with 2 neurons and 5 inputs (5 inputs because this is the output from the previous Layer1),                                   
layer2 = Layer_Dense(1,5)


layer1.forward(X)

print(layer1.output, layer1.output.shape)

layer2.forward(layer1.output)

print('-------------------------------------')
print(layer2.output, layer2.output.shape)