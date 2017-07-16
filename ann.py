import math
from numpy import dot, subtract

# constants
alpha = 1 # size of gradient descent step
beta = 1/255 # sharpness of sigmoid
delta = 0.1 # temporary weight adjustment to approximate derivative

# components
L = [] # layers
W = [] # weights
b = [] # biases

# Set up layers, weights, and biases; init everything to 0
def init(layer_sizes):
    clear()
    # input layer has no weights/biases associated with it
    L.append([])
    for j in range(0, layer_sizes[0]):
        L[0].append(0)
    # set up remaining layers with weights & biases
    for i in range(1, len(layer_sizes)):
        L.append([])
        b.append([])
        W.append([])
        for j in range(0, layer_sizes[i]):
            L[i].append(0)
            b[i-1].append(0)
            W[i-1].append([])
            for k in range(0, layer_sizes[i-1]):
                W[i-1][j].append(0)

# Clear ANN
def clear():
    global L, W, b
    L = []
    W = []
    b = []

# Sigmoid function
#
# May cause overflow error if beta*z is too large
# Typically, making beta 1/x_i (normalizing the inputs), should prevent this
def sigmoid(z):
    return 1 / (1 + math.exp(-beta*z))

# Activation function
#
# Reprogram as desired:
# ann.activation = my_func (or use one of the functions in ann.py)
def activation(z):
    return sigmoid(z)

# Calculates output for input x
#
# Fills L
def output(x):
    #global L
    if(len(x) != len(L[0])):
        return "Error: Input vector length mismatch"
    L[0] = x
    for i in range(0,len(L)-1):
        for j in range(0,len(L[i+1])):
            #print("i,j: " + str(i) + "," + str(j))
            L[i+1][j] = activation(dot(W[i][j],L[i])+b[i][j]) 
    return L[len(L)-1]

# Calculates expected output for an input index
#
# Reprogram as necessary by assigning your own function to expected:
# ann.expected = my_func
def expected():
    return 0

# Calculates total error
#
# x = input vector
#
# Compares full output vector with expected vector, runs through whole ANN
#
# Inefficient! Avoid frequent use
def total_error(x):
    if(len(x) != len(L[0])):
        return "Error: Input vector length mismatch"
    diff = subtract(expected(),output(x))
    return dot(diff,diff)

# Calculates error focused on a single node
#
# v = vector/layer of values to be processed
# w = weights through which v will be processed (weights pointing to node)
# b = bias of node
# ex = expected value of node
def node_error(v, w, b, ex):
    return (activation(dot(w,v)+b)-ex)**2

# Optimized node error function
#
# Actually, it's not that this function is optimized, it simply takes a
# known result as an argument, instead of evaluating a dot product of
# vectors (expensive!)
#
# The calling function should intelligently calculate the result, by
# calculating the dot product once, and adjusting it for each weight
# adjustment, instead of letting node_error recalculate the full dot
# product each time
#
# res = resultant value at node
def optimized_node_error(res, ex):
    return (activation(res)-ex)**2

# Brute training method with gradient descent
#
# Adjusts a component and evaluates change in total error
#
# Way too many calculations, horrendously inefficient (runs output once for
# each weight/bias)
def brute_train(x):
    err = total_error(x)
    w_deltas = []
    b_deltas = []
    for i in range(0,len(W)):
        w_deltas.append([])
        for j in range(0,len(W[i])):
            w_deltas[i].append([])
            for k in range(0,len(W[i][j])):
                W[i][j][k] += 0.1 # adjust weight
                w_deltas[i][j].append(total_error(x)-err) # calculate
                # and save change in error
                W[i][j][k] -= 0.1
    for i in range(0,len(W)):
        b_deltas.append([])
        for j in range(0,len(W[i])):
            b[i][j] += 0.1
            b_deltas[i].append(total_error(x)-err)
            b[i][j] -= 0.1
    for i in range(0,len(W)):
        for j in range(0,len(W[i])):
            for k in range(0,len(W[i][j])):
                W[i][j][k] -= w_deltas[i][j][k]*alpha
            b[i][j] -= b_deltas[i][j]*alpha

# Gradient descent with backpropagation
#
# Adjusts an element and calculates change in error, considering only expected
# value of node weight points to
#
# Not perfectly implemented
# Backpropagation not implemented correctly for multiple layers
# Total error is not considered, only error on individual nodes (this can be
# be efficiently achieved using backprop)
def train(x):
    output(x) # fill L
    L[-1] = expected() # assign expected outputs
    w_deltas = []
    b_deltas = []
    for i in range(0, len(W)):
        w_deltas.append([])
        for j in range(0, len(W[i])):
            w_deltas[i].append([])
            dot_prod = dot(W[i][j], L[i])
            for k in range(0, len(W[i][j])):
                err = optimized_node_error(dot_prod + b[i][j], L[i+1][j])
                dot_prod -= W[i][j][k]*L[i][k]
                W[i][j][k] += 0.1
                dot_prod += W[i][j][k]*L[i][k]
                w_deltas[i][j].append(optimized_node_error(dot_prod + b[i][j],
                L[i+1][j]) - err)
                dot_prod -= W[i][j][k]*L[i][k]
                W[i][j][k] -= 0.1
                dot_prod += W[i][j][k]*L[i][k]
    for i in range(0, len(W)):
        b_deltas.append([])
        for j in range(0, len(W[i])):
            dot_prod = dot(W[i][j], L[i])
            err = optimized_node_error(dot_prod + b[i][j], L[i+1][j])
            b[i][j] += 0.1
            b_deltas[i].append(optimized_node_error(dot_prod + b[i][j], L[i+1][j])
                               - err)
            b[i][j] -= 0.1
    for i in range(0,len(W)):
        for j in range(0,len(W[i])):
            for k in range(0,len(W[i][j])):
                W[i][j][k] -= w_deltas[i][j][k]*alpha
            b[i][j] -= b_deltas[i][j]*alpha
                

# TODO:
# See about calculating derivatives analytically, as opposed to approximating by
# running net
#
# Read up on backpropagation, check if implemented correctly
#
# Implement softmax
#
# Test simple linear regression
#
# Try all these out using Tensorflow!
# 
                
