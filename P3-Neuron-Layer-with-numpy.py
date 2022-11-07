'''
Creates a simple layer of 3 neurons, with 4 inputs, with numpy dor product.
'''

import numpy as np 

#shap: 1*4
inputs = [1.0, 2.0, 3.0, 2.5]

#shap: 4*3
weights = [[0.2,  0.5,  -0.26],
           [0.8, -0.91, -0.27],
           [-0.5, 0.26,  0.17],
           [1.0, -0.5,   0.87]]

biases = [2.0, 3.0, 0.5]


output = np.dot(inputs, weights) + biases


print(output)
#shap: 1*3
#[4.8   1.21  2.385]