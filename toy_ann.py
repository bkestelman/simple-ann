import numpy, math

w = numpy.array([0.,0.])
b = 0
inputs = [numpy.array([0,0]), numpy.array([1,0]), numpy.array([0,1]), numpy.array([1,1])]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def output(x):
    return sigmoid(w.dot(x)+b)

def expected(x):
    return x[0]==1 and x[1]==1

def error(x):
    return (output(x)-expected(x))**2

def train(x):
    global w
    global b
    err = error(x)
    w[0] += 0.1
    ddw0 = error(x)-err
    w[0] -= 0.1
    w[1] += 0.1
    ddw1 = error(x)-err
    w[1] -= 0.1
    b += 0.1
    ddb = error(x)-err
    b -= 0.1
    w[0] -= ddw0*0.1
    w[1] -= ddw1*0.1
    b -= ddb*0.1
