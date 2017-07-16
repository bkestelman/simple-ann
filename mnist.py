train_labels = open('train-labels.idx1-ubyte', 'rb')
train_images = open('train-images.idx3-ubyte', 'rb')
test_labels = open('t10k-labels.idx1-ubyte', 'rb')
test_images = open('t10k-images.idx3-ubyte', 'rb')

labels_read = 0
images_read = 0
    
def init():
    train_labels.seek(0)
    train_images.seek(0)
    test_labels.seek(0)
    test_images.seek(0)
    labels_read = 0
    images_read = 0
    # verify magic number and prepare to read
    if int.from_bytes(train_labels.read(4), byteorder='big') != 2049:
        return "Error: Wrong magic number for train_labels"
    if int.from_bytes(train_images.read(4), byteorder='big') != 2051:
        return "Error: Wrong magic number for train_images"
    if int.from_bytes(test_labels.read(4), byteorder='big') != 2049:
        return "Error: Wrong magic number for test_labels"
    if int.from_bytes(test_images.read(4), byteorder='big') != 2051:
        return "Error: Wrong magic number for test_images"
    train_labels.read(4) # read number of items (60000)
    train_images.read(12) # read number of items and dimensions (60000, 28, 28)
    test_labels.read(4) # (10000)
    test_images.read(12) # (10000, 28, 28)

# reads a label and returns as int
def read_label():
    global labels_read
    labels_read += 1
    if labels_read < 60000:
        return int.from_bytes(train_labels.read(1), byteorder='big')
    else:
        return int.from_bytes(test_labels.read(1), byteorder='big')

# reads a label and returns as 1D array of 28*28 ints
def read_image():
    global images_read
    images_read += 1
    ret = []
    for i in range(0, 28*28):
        if images_read < 60000:
            ret.append(int.from_bytes(train_images.read(1), byteorder='big'))
        else:
            ret.append(int.from_bytes(test_images.read(1), byteorder='big'))
    return ret

# function to pass to ann.expected
# ann.expected = mnist.expected
def expected():
    global labels_read
    one_hot = [0,0,0,0,0,0,0,0,0,0]
    one_hot[read_label()] = 1
##    labels_read -= 1
##    if labels_read < 60000:
##        train_labels.seek(-1, 1) # go back (expected may be called multiple
##        # times, expecting the same label; read should only move forward in
##        # file when user explicitly calls read_label())
##    else:
##        test_labels.seek(-1, 1)
    return one_hot
