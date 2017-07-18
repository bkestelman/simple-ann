# simple-ann

Simple artificial neural net in python, using MNIST dataset of hand-written digits (http://yann.lecun.com/exdb/mnist/)

## Simple Usage

Run 
```
$ python -i mnist_setup.py # launch a python shell after running mnist_setup.py
>>> for i in range(1, 100):
...    ann.train(mnist.read_image()) # run image through ann, applying gradient descent
...
>>> o = ann.output(mnist.read_image()) # run the next image through ann, without applying gradient descent
>>> mnist.read_label() # read answer to last image read
1
>>> argmax(o) # read answer produced by ann (ann.output() produces an array with "probabilities" for each answer, so the index of the max element is the most probable answer)
1
>>> o = ann.output(mnist.read_image())
>>> mnist.read_label() 
5
>>> argmax(o)
6
>>> mnist.read_label()
7
>>> argmax(o)
7
>>> correct = 0
>>> for i in range(1, 100):
...    if argmax(ann.output(mnist.read_image())) == mnist.read_label():
...    correct += 1
>>> correct
52 # after training on only 100 images, ann performs at roughly 52% accuracy
# training on more images produces better results, but takes significantly longer
```

## Code Organization

All the ANN stuff is in ann.py

Reading images and labels from MNIST data is in mnist.py

Short initialization code is run in setup_mnist.py
