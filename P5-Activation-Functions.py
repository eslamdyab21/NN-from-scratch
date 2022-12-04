import numpy as np 



class Layer_Dense:
    def __init__(self, n_neurons, n_inputs, activation_type):
        #intialize the weights randmoly
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        

        #intialize the biases with zeros
        self.biases = np.zeros((n_neurons,1))


        self.activation_type = activation_type
        self.activation = None

        # print(self.weights, self.weights.shape)
        # print('-----------------------------------------')


    def forward(self, inputs):
        #forword path
        self.z = np.dot(self.weights, inputs) + self.biases
        
        #activation
        if self.activation_type == 'ReLU':
            self.activation = self.Activation_ReLU(self.z)
            # print('self.zzzzzzz', self.z)
            # print('self.activation000', self.activation)

        elif self.activation_type == 'Softmax':
            self.activation = self.Activation_Softmax(self.z)


    def Activation_ReLU(self,inputs):
        return np.maximum(0, inputs)


    def Activation_Softmax(self,inputs):
        probabilities = np.exp(inputs) / sum(np.exp(inputs))
        return probabilities




X = np.linspace(0-5,5,11).reshape(11,1)

layer1 = Layer_Dense(3,11, 'ReLU')
layer1.forward(X)

print(X)
print('-------------------------------------')
print(layer1.z)
print('-------------------------------------')
print(layer1.activation)
print('-------------------------------------')