import numpy as np 
import nnfs
from nnfs.datasets import vertical_data





class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation_type):
        #intialize the weights randmoly
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        #intialize the biases with zeros
        self.biases = np.zeros((1, n_neurons))

        self.activation_type = activation_type


    def forward(self, inputs):
        #forword path
        self.output = np.dot(inputs, self.weights) + self.biases
        
        #activation
        if self.activation_type == 'ReLU':
            self.output = self.Activation_ReLU(self.output)

        elif self.activation_type == 'Softmax':
            self.output = self.Activation_Softmax(self.output)


    def Activation_ReLU(self,inputs):
        return np.maximum(0, inputs)


    def Activation_Softmax(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

        
class Train_Model:
    def __init__(self, model_architecture, X, Y, learning_rate=0.001):
        self.model_architecture = model_architecture
        self.X = X
        self.Y = Y

        self.Train()

    def forward(self):
        #forword path
        self.model_architecture[0].forward(X)
        self.model_architecture[1].forward(self.model_architecture[0].output)
        print(self.model_architecture[1].output)


    def Train(self):
        self.forward()



nnfs.init()        
X, y = vertical_data(samples=10, classes=3)
'''
[[ 0.17640524  0.51440436]
 [ 0.04001572  0.64542735]
 [ 0.0978738   0.57610375]
 [ 0.22408931  0.5121675 ]
 [ 0.1867558   0.5443863 ]
 [-0.09772779  0.53336746]
 [ 0.09500884  0.6494079 ]
 [-0.01513572  0.47948417]
 [-0.01032189  0.53130674]
 [ 0.04105985  0.41459042]

 [ 0.07803437  0.51549476]
 [ 0.3986952   0.5378162 ]
 [ 0.41977698  0.4112214 ]
 [ 0.25911683  0.30192035]
 [ 0.5603088   0.46520877]
 [ 0.18789677  0.5156349 ]
 [ 0.3379092   0.62302905]
 [ 0.31461495  0.620238  ]
 [ 0.48661125  0.46126732]
 [ 0.48026922  0.46976972]
 
 [ 0.5618114   0.41045335]
 [ 0.5246649   0.53869027]
 [ 0.49603966  0.44891948]
 [ 0.8617442   0.3819368 ]
 [ 0.61570144  0.49718177]
 [ 0.62285924  0.5428332 ]
 [ 0.54138714  0.5066517 ]
 [ 0.7444157   0.5302472 ]
 [ 0.5052769   0.43656778]
 [ 0.64539266  0.4637259 ]]
'''


model = [Layer_Dense(2,3,'ReLU'), Layer_Dense(3, 3,'Softmax')]
Train_Model(model,X, y)
