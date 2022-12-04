'''
Creates a simple layer of 3 neurons, with 4 inputs, with numpy dor product.
'''

import numpy as np 


inputs = [1.0, 2.0, 3.0, 2.5]
#shap: 4*1 (col vector)
inputs = np.array(inputs).T
inputs = inputs.reshape(4,1)

#shap: 3*4
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
weights = np.array(weights)


biases = [2.0, 3.0, 0.5]
#shape: 3*1 (col vector)
biases = np.array(biases).T
biases = biases.reshape(3,1)


# 3x4 * 4x1 + 3x1
# 3x1 + 3x1
# 3x1
# col vector
output = np.dot(weights, inputs) + biases


print(output, output.shape)
#shap: 3x1
#[4.8   1.21  2.385].T