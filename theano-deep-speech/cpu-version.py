__author__ = 'pat'
import numpy as np
import math
import time
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cPickle as pickle

import brnn as nn


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
    # return [blank] + sum([[i, blank] for i in l], [])

DEBUG = False
TEST = False

alphabet = np.arange(29) #[a,....,z, space, ', -]


# Load samples
import os
samples = []
directory = 'pickled'
files = [os.path.join(directory, x) for x in os.listdir(directory)]
#files = files[:25]

for f in files:
    submission = pickle.load(open(f, 'rb'))
    print f

    for sample in submission:
        label_len = len(sample[0])
        label_prime_len = len(make_l_prime(str_to_seq(F(sample[0], len(alphabet))), len(alphabet)))
        num_buckets = np.shape(sample[1])[0]
        if label_len < 37 and label_prime_len <= num_buckets:# and (float(num_buckets) / float(label_prime_len) < 3.0):
            window_size = 1 # 1 frame of context on each side
            windowed = []
            for i in np.arange(window_size, len(sample[1])-window_size):
                windowed.append(np.concatenate((sample[1][i-1], sample[1][i], sample[1][i+1])))
        
            samples.append( (make_l_prime(str_to_seq(F(sample[0], len(alphabet))), len(alphabet)), windowed) )
        
            #samples_init.append(sample)

visual_sample = samples[0]
print 'Shape of first input:', np.shape(visual_sample[1])
print seq_to_str(visual_sample[0])
    

duration = time.time()
net = nn.Network()
#network = net.create_network(np.shape(samples[0][1])[1]*3, len(alphabet)+1)        #x3 for the window
network = net.create_network(240, len(alphabet)+1)        #x3 for the window
print 'built network - num samples:', len(samples), '\tDuration: %f' % (time.time()-duration)


rng = np.random.RandomState(1234)

try:
    for epoch in np.arange(100000):
        rng.shuffle(samples)

        avg_error = 0.0

        duration = time.time()

        for s in samples:
        
            cst, pred = network.trainer(s[1], s[0])
            avg_error += cst

        if math.isnan(avg_error) or math.isinf(avg_error):

            for x in np.arange(10):
                [' ' for y in range(x)]
                print [' ' for y in range(x)], 'Hit nan/inf'

                raw_input('hit NAN')

        print 'Epoch:', epoch, '\tAvg Error:', avg_error / len(samples), '\tin %0.3fs' %(time.time()-duration), '\t' + str(len(samples)), 'samples', '\tSamples/sec: %0.3f' % (len(samples)/(time.time()-duration)), '\tApprox. Speed: %0.3fx' % ((len(samples) * 5) / (time.time()-duration)), 'real-time'

        if epoch % 1 == 0:
            net.dump_network('cpu/' + str(epoch) + '.pkl')

            pred = network.tester(visual_sample[1])[0]
            
            print seq_to_str(visual_sample[0]), '||', seq_to_str([np.argmax(x) for x in pred])

           
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
