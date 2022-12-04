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

        

class Train_Model:
    def __init__(self, model_architecture, X, Y, learning_rate=0.1, epochs=100):
        self.model_architecture = model_architecture
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.Train()


    def Train(self):
        # self.model_architecture[0].weights = self.model_architecture[0].weights*0
        # print(self.model_architecture[0].weights)

        for epoch in range(self.epochs):
            Y_pred = self.forward(self.X)
            #self.backward(Y_pred)
            #self.update_params()

            '''
            if epoch % 10 == 0:
                print("epoch: ", epoch)
                predictions = self.get_predictions(Y_pred)
                print(self.get_accuracy(predictions, self.Y))
            '''



    def forward(self, X):
        self.model_architecture[0].forward(X)
        self.model_architecture[1].forward(self.model_architecture[0].activation)
        self.model_architecture[2].forward(self.model_architecture[1].activation)

        Y_pred = self.model_architecture[2].activation

        return Y_pred


    

model = [Layer_Dense(10,784,'ReLU'), Layer_Dense(20,10,'ReLU'), Layer_Dense(10, 20,'Softmax')]
#Train_Model(model,X_train, Y_train, epochs=200, learning_rate=0.1)


'''
         
 x       
 x             
 .                  o
 .           o      o
 x           o      o

 X           h0     h1
784*41000  10*784  10*10
              10*41000
'''

'''     
 x       
 x             
 .                  o        o
 .           o      o        o
 x           o      o        o

 X           h0     h1       h2
784*41000  10*784  10*10    10*10
              10*41000  10*41000   10*41000
'''