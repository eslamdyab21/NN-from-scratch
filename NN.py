import numpy as np 
from matplotlib import pyplot as plt


class Layer_Dense:
    def __init__(self, n_neurons, n_inputs, activation_type):
        #intialize the weights randmoly
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.dw = 0.10 * np.random.randn(n_neurons, n_inputs)

        

        #intialize the biases with zeros
        self.biases = np.zeros((n_neurons,1))
        self.db = np.zeros((n_neurons,1))


        self.activation_type = activation_type
        self.activation = None

        # print(self.weights, self.weights.shape)
        # print(self.dw, self.dw.shape)
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

        #self.Train()


    def Train(self):
        # self.model_architecture[0].weights = self.model_architecture[0].weights*0
        # print(self.model_architecture[0].weights)

        for epoch in range(self.epochs):
            Y_pred = self.forward(self.X)
            self.backward(Y_pred)
            self.update_params()

            if epoch % 10 == 0:
                print("epoch: ", epoch)
                predictions = self.one_hot_decode(Y_pred)
                print('acc', self.get_accuracy(predictions, self.Y))
                loss = self.loss(Y_pred, self.one_hot(self.Y))
                print('loss', loss)
                print('----------------------------------------------')



    def forward(self, X):
        #forword path
        layers_num = len(self.model_architecture)

        self.model_architecture[0].forward(X)
        for layer in range(1,layers_num):
            self.model_architecture[layer].forward(self.model_architecture[layer-1].activation)

        Y_pred = self.model_architecture[-1].activation

        return Y_pred


    def backward(self, Y_pred):
        m = self.X.shape[1]
        layers_num = len(self.model_architecture)

        one_hot_Y = self.one_hot(self.Y)
        A_prev = self.model_architecture[-2].activation

        dZL = self.loss_derivative(Y_pred, one_hot_Y)
        # print('ypreddddddd', Y_pred[:,0], Y_pred.shape)
        # print('oneeeeehott', one_hot_Y[:,0], one_hot_Y.shape)
        # print('DZLLLLLLLLL', dZL[:,0], dZL.shape)
        dWL = (1 / m) * (dZL.dot(A_prev.T))
        dbL = (1 / m) * np.sum(dZL)
        # print('A1', A1)
        # print('dZL.dot(A1.T)', dZL.dot(A1.T))
        # print('dwwwwwllll', dWL, dWL.shape)
        # print('dbblllllll', dbL, dbL.shape)

        self.model_architecture[-1].dw = dWL
        self.model_architecture[-1].db = dbL

        # print('*************************')
        # print('dWL' ,dWL.shape)
        # print('dbL', dbL.shape)
        # print('self.model_architecture[-1].activation', self.model_architecture[-1].activation.shape)

        dZ_prev = dZL
        w_prev = self.model_architecture[-1].weights
        for layer in range(2,layers_num + 1):
            layer = - layer
            Z = self.model_architecture[layer].z

            if abs(layer-1) <=  layers_num:
                A_af = self.model_architecture[layer-1].activation
            else:
                A_af = self.X


            dZ = w_prev.T.dot(dZ_prev) * self.ReLU_deriv(Z)
            dW = (1 / m) * (dZ.dot(A_af.T))
            db = (1 / m) * np.sum(dZ)

            dZ_prev = dZ
            w_prev = self.model_architecture[layer].weights
            
            self.model_architecture[layer].dw = dW
            self.model_architecture[layer].db = db

            # print('*************************')
            # print('layer', layer)
            # print('dW', dW.shape)
            # print('db', db.shape)
            # print('self.model_architecture[layer].activation', self.model_architecture[layer].activation.shape)


    def update_params(self):
        layers_num = len(self.model_architecture)
        for layer in range(layers_num):
            self.model_architecture[layer].weights = self.model_architecture[layer].weights - self.learning_rate*self.model_architecture[layer].dw
            self.model_architecture[layer].biases = self.model_architecture[layer].biases - self.learning_rate*self.model_architecture[layer].db

        #     print('*************************')
        #     print('W',self.model_architecture[layer].dw, self.model_architecture[layer].dw.shape)
        #     print('*************************')
        #     print('b', self.model_architecture[layer].db, self.model_architecture[layer].db.shape)
        # print('================================')

    def one_hot_decode(self, Aactivation_softmax):
        return np.argmax(Aactivation_softmax, 0)


    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size


    def make_predictions(self, X):
        Aactivation_softmax = self.forward(X)
        predictions = self.one_hot_decode(Aactivation_softmax)
        return predictions


    def predict_probability(self, image):
        prediction = self.forward(image)
        print("Prediction: ", prediction)
        
        image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.show()
    

    def predict_label(self, image):
        prediction = self.make_predictions(image)
        print("Prediction: ", prediction)
        
        image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.show()



    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        
        return one_hot_Y
    
    def loss_derivative(self, output_activations, y):
        return (output_activations-y)


    def loss(self, output_activations, y):
        return np.mean((output_activations-y)**2)


    def ReLU_deriv(self, Z):
        return Z > 0

    

