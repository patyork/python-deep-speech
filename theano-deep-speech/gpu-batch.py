__author__ = 'pat'
import numpy as np
import math
import time
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cPickle as pickle
import theano

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
    
    
def log_it(f, epoch=None, error=None, etime=None, samples=None, nan=False, etc=None):
    s5 = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
    if epoch is not None:
        message = time.strftime('%H:%M:%S') + '\tEpoch: ' + str(epoch) + '\tAvg. Error: ' + str(error / samples) + '\tin ' + str(etime) + 's\t' +str(samples) + ' samples\tSamples/sec: ' + str(samples/etime) + '\tApprox. Speed: ' + str(samples * 5 / etime) + 'x real-time'
        print '\n', message, '\n'
        f.write(message.replace('\t', s5) + '<br />')
    elif nan:
        print time.strftime('%H:%M:%S') + '\n========\nHIT NAN=======\nInvalidating Epoch\n===========\n'
        f.write('<br />' + time.strftime('%H:%M:%S') + '\n========\nHIT NAN=======\nInvalidating Epoch\n===========\n\n<br />')
        
    else:
        print etc
        f.write(time.strftime('%H:%M:%S') +s5+ etc.replace('\n', '<br />').replace('\t', s5) + '<br />')


alphabet = np.arange(29) #[a,....,z, space, ', -]


# Load samples
f = open('win3_l35.pkl', 'rb')
samples = pickle.load(f)
f.close()

#print [len(s[0]) for s in samples]

samples_sorted = sorted(samples, key=lambda s: len(s[0]))

lens = [len(s[0]) for s in samples_sorted]
#print lens

lp_lens = [key for key, _ in groupby(lens)]
#print lp_lens
#print

batch_dict = {}
for length in lp_lens:
    # Get samples of length
    sample_of_len = []
    label_of_len = []
    for s in samples:
        if len(s[0]) == length:
            sample_of_len.append(np.asarray(s[1], dtype=theano.config.floatX))
            label_of_len.append(np.asarray(s[0], dtype='int32'))

    pad_to = np.max([s.shape[0] for s in sample_of_len])
    padded = []
    for s in sample_of_len:

        pad = np.zeros((pad_to-s.shape[0], s.shape[1]))
        padded.append( np.concatenate((s, pad), axis=0) )

    #print type(padded), type(padded[0])
    #print type(label_of_len), type(label_of_len[0])

    #for s in padded:
    #    print s.shape

    batch_dict[length] = (np.asarray(label_of_len, dtype='int32'), np.asarray(padded, dtype=theano.config.floatX))


visual_sample = samples[0]
visual_sample2 = samples[1]
print 'Shape of first input:', np.shape(visual_sample[1])
print seq_to_str(visual_sample[0])
print seq_to_str(visual_sample2[0])


# PARAMETERS
epoch_size = 10        # mini-batches per epoch (model is stored at the end of this many mini-batches)
batch_size = 100        # mini-batch size

# HYPER-PARAMETERS
learning_rate = .01
momentum_coefficient = .25


net = nn.Network()
rng = np.random.RandomState(int(time.time()))

# create network
try:
    last_good = -1
    restart = False
    #log_file = file('/var/www/html/status.html', 'a')
    log_file = file('status.html', 'a')
    duration = time.time()
    if last_good == -1:
        network = net.create_network(560, len(alphabet)+1, .0005, .75)        #x3 for the window
        log_it(log_file, etc='created Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
    else:
        picklePath = 'saved_models_batch/' + str(last_good) + '.pkl'
        print 'loading from', picklePath
        network = net.load_network(picklePath, 560, len(alphabet)+1, .0005, .75)        #x3 for the window
        log_it(log_file, etc='loaded Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
    log_file.close()

    # Start a new Epoch
    for epoch in np.arange(last_good+1, 100000):

        # For each desired mini-batch
        for minibatch in np.arange(epoch_size):

            sequence_length_index = rng.randint(0, 1)#len(lp_lens))                                # randomly select a bucket of samples to create a mini-batch from
            sequence_length = lp_lens[sequence_length_index]

            num_samples_in_selected = batch_dict[sequence_length][1].shape[0]    # get the number of available samples in the bucket
            print batch_dict[sequence_length][1].shape
            
            if num_samples_in_selected < batch_size:
                minibatch_start = 0
                minibatch_end = num_samples_in_selected
            else:
                minibatch_start = rng.randint(0, len(lp_lens))
                minibatch_end = minibatch_start + batch_size

            cost, pred = network.trainer(batch_dict[sequence_length][1][minibatch_start:minibatch_end, :, :], batch_dict[sequence_length][0][minibatch_start:minibatch_end, :])
            
            print cost

except KeyboardInterrupt:
    pass

# shuffle lp_lens

# take lp_lens[0] as length

    # random from [0, batch_dict[length]-batch_size]

    # batch train on the random slice

    # every Xth training batch, save model, shuffle


