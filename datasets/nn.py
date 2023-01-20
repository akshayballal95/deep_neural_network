import numpy as np
import pandas as pd

print("Hello")

layer_dims = [12288, 20, 7, 5, 1]


def load_data():
    data = pd.read_csv("training_set.csv")
    x_train = data.drop(columns=['y'])
    x_train = x_train.to_numpy().T/255.0
    y_train = data.get(['y'])
    y_train = y_train.to_numpy().T

    return (x_train, y_train)


def relu(z):
    A = np.maximum(0,z)
    cache = z
    return A, cache


def relu_prime(z):
    z = z.copy()
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def relu_activation(z):
    return (z*(z > 0), z)


def relu_backward(da, cache):
    Z = cache
    dZ = np.array(da, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid(z):
    A = 1/(1+np.exp(-z))
    cache = z
    
    return A, cache


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def sigmoid_activation(z):
    return (sigmoid(z), z)


def sigmoid_backward(da, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = da * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape ==
                (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(a, w, b):

    z = np.dot(w, a) + b
    cache = (a, w, b)

    return (z, cache)


def linear_forward_activation(a_prev, w, b, activation):
    if activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    if activation == "relu":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)
    
    cache = (linear_cache, activation_cache)
    return (a, cache)


def l_model_forward(x, parameters):

    number_of_layers = len(layer_dims)-1
    a = x.copy()
    caches = {}

    for l in range(1, number_of_layers):
        a_prev = a
        weight_string = "W"+str(l)
        bias_string = "b"+str(l)
        w = parameters[weight_string]
        b = parameters[bias_string]

        a, cache = linear_forward_activation(a_prev, w, b, "relu")
        caches[str(l)] = cache

    weight_string = "W"+str(number_of_layers)
    bias_string = "b"+str(number_of_layers)

    w = parameters[weight_string]
    b = parameters[bias_string]

    al, cache = linear_forward_activation(a, w, b, "sigmoid")
    caches[str(number_of_layers)] = cache

    return (al, caches)

def cost(al, y):
    m = y.shape[1]
    cost = -(1/m)*(np.dot(y, np.log(al.T)) + np.dot(1-y, np.log((1-al)).T))
    return np.sum(cost)

def linear_backward(dz, linear_cache):
    a_prev, w, b = linear_cache
    m = a_prev.shape[1]
    dw = (1.0/m)*(np.dot(dz, a_prev.T))
    db = (1.0/m)*np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(w.T, dz)

    return (da_prev, dw, db)


def linear_backward_activation(da, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dz = relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    if activation == "sigmoid":
        dz = sigmoid_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    return (da_prev, dw, db)



def l_model_backward(al, y, caches):
    grads = {}
    num_layers = len(layer_dims)-1
    dal = -(y/al - (1-y)/(1-al))

    current_cache = caches[str(num_layers)]
    da_prev, dw, db = linear_backward_activation(dal, current_cache, "sigmoid")

    weight_string = "dW"+str(num_layers)
    bias_string = "db"+str(num_layers)
    activation_string = "dA"+str(num_layers)

    grads[weight_string] = dw
    grads[bias_string] = db
    grads[activation_string] = da_prev

    for l in reversed(range(1, num_layers)):
        current_cache = caches[str(l)]
        da_prev, dw, db = linear_backward_activation(
            da_prev, current_cache, "relu")

        weight_string = "dW"+str(l)
        bias_string = "db"+str(l)
        activation_string = "dA"+str(l)

        grads[weight_string] = dw
        grads[bias_string] = db
        grads[activation_string] = da_prev


    return grads

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    num_of_layers = len(layer_dims)-1

    for l in range(1,num_of_layers+1):
        weight_string_grad = "dW"+str(l)
        bias_string_grad = "db"+str(l)
        weight_string = "W"+str(l)
        bias_string = "b"+str(l)

        parameters[weight_string] = parameters[weight_string] - learning_rate*grads[weight_string_grad]
        parameters[bias_string] = parameters[bias_string] - learning_rate*grads[bias_string_grad]

    return parameters



np.random.seed(1)

x_train, y_train = load_data()
parameters = initialize_parameters(layer_dims)

print(parameters)
# print(x_train)
# print(y_train)


for i in range(0,2500):
    al, caches = l_model_forward(x_train, parameters)
    costy = cost(al,y_train)
    grads = l_model_backward(al, y_train, caches)
    parameters = update_parameters(parameters, grads, 0.0075)
    print("Epoch: {}/ 2500     Cost: {}".format(i, costy))
