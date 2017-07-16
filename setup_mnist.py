import ann
import mnist
from numpy import argmax

mnist.init()
ann.init([784,10])
ann.expected = mnist.expected
