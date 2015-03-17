__author__ = 'pat'
import numpy as np
import math
import time
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cPickle as pickle
import brnngpu as nn
import theano

#THEANO_FLAGS='device=cpu,floatX=float32'


def str_to_seq(str):
    seq = []
    for c in str:
        val = ord(c)
        if val==32:
            val = 26
        elif val==39:
            val=27
        elif val==45:
            val=28
        else:
            val-=97
        seq.append(val)
    return seq


def seq_to_str(seq):
    str = ''
    for elem in seq:
        if elem==26:
            str += ' '
        elif elem==27:
            str += '\''
        elif elem==28:
            str += '-'
        elif elem==29:
            pass
        else:
            str += chr(elem+97)
    return str


# Remove consecutive symbols and blanks
def F(pi, blank):
    return [a for a in [key for key, _ in groupby(pi)] if a != blank]


# Insert blanks between unique symbols, and at the beginning and end
def make_l_prime(l, blank):
    result = [blank] * (len(l) * 2 + 1)
    result[1::2] = l
    return result


alphabet = np.arange(29) #[a,....,z, space, ', -]


# Load samples
import os
samples = []
directory = 'pickled'
files = [os.path.join(directory, x) for x in os.listdir(directory)]
files = files[:100]


for f in files:
    submission = pickle.load(open(f, 'rb'))
    print f

    for sample in submission:
        label_len = len(sample[0])
        label_prime_len = len(make_l_prime(str_to_seq(F(sample[0], len(alphabet))), len(alphabet)))
        num_buckets = np.shape(sample[1])[0]
        if label_len < 37 and label_prime_len <= num_buckets:# and (float(num_buckets) / float(label_prime_len) < 3.0):
            samples.append(sample)


def generate_shared(samples, blank):
    maximum_x = np.max([s[1].shape[0] for s in samples])
    maximum_y = np.max([len(s[0]) for s in samples])*2+1

    win_x = []
    y_data = []
    for s in samples:
        # Window and send
        window_size = 1 # 1 frame of context on each side
        windowed = []
        for i in np.arange(window_size, len(s[1])-window_size):
            windowed.append(np.concatenate((s[1][i-1], s[1][i], s[1][i+1])))

        for j in np.arange(len(s[1]), maximum_x):
            windowed.append(np.zeros((240)))

        windowed = np.asarray(windowed, dtype=theano.config.floatX)
        win_x.append(windowed)


        l_p = make_l_prime(str_to_seq(F(s[0], len(alphabet))), len(alphabet))
        l_p += [blank for x in np.arange(maximum_y-len(l_p))]

        y_data.append(np.asarray(l_p, dtype='int32'))

    x_data = np.rollaxis(np.dstack(win_x), -1)
    y_data = np.asarray(y_data, dtype='int32')

    print x_data.shape, x_data.dtype, x_data[0].shape
    print y_data.shape, y_data.dtype, y_data[0].shape

    shared_x_data = theano.shared(x_data, borrow=True)
    shared_y_data = theano.shared(np.asarray(y_data), borrow=True)

    return shared_x_data, theano.tensor.cast(shared_y_data, 'int32')


shared_x, shared_y = generate_shared(samples, len(alphabet))


#network = nn.BRNN(np.shape(samples[0][1])[1], len(alphabet)+1)        #x3 for the window
duration = time.time()
network = nn.BRNN(240, len(alphabet)+1, shared_x, shared_y)        #x3 for the window
print 'built network - num samples:', len(samples), '\tBuild Time: %fs' % (time.time()-duration)


for asfdasdf in np.arange(5):
    for ndexx in np.arange(math.floor(1001/1000)):
        duration = time.time()
        sOut = network.debugTest(ndexx)

        print 'Shape: ', sOut[0].shape, '\tValue: ', sOut, '\tDuration: %f' % (time.time()-duration)
    print '\n\n'

raw_input()


minibatches = len(samples)

rng = np.random.RandomState(1234)
try:
    for epoch in np.arange(1):
        error = 0.0
        duration = time.time()
        for minibatch_index in np.arange(minibatches):
            cst, pred = network.trainer(minibatch_index)
            error += cst

        print 'Epoch %i:\tAverage Error: %f\tDuration: %f' % (epoch, error/minibatches, time.time()-duration)

except KeyboardInterrupt:
    for s in samples:
        # Window and send
        window_size = 1 # 1 frame of context on each side
        windowed = []
        for i in np.arange(window_size, len(s[1])-window_size):
            windowed.append(np.concatenate((s[1][i-1], s[1][i], s[1][i+1])))
        
        windowed = np.asarray(windowed)
        #windowed=s[1]

        print s[0], '||', seq_to_str([np.argmax(x) for x in network.tester(windowed)[0]])

raw_input('asdgf')
