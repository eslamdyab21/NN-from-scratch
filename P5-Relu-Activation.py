import numpy as np 


X = np.linspace(0-5,5,11).reshape(11,1)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #intialize the weights randmoly
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        #intialize the biases with zeros
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        #forword path
        self.output = np.dot(inputs, self.weights) + self.biases



class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



layer1 = Layer_Dense(1,1)
activation1 = Activation_ReLU()

layer1.forward(X)

activation1.forward(layer1.output)


print(X)
print('-------------------------------------')
print(layer1.output)
print('-------------------------------------')
print(activation1.output)
print('-------------------------------------')