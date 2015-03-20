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
    
    
def log_it(f, epoch=None, error=None, etime=None, samples=None, saved_to=None, nan=False):
    if nan is False:
        s5 = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        message = time.strftime('%H:%M:%S') + 'Epoch: ' + str(epoch) + '\tAvg. Error: ' + str(error / samples) + '\tin ' + str(etime) + 's\t' +str(samples) + ' samples\tSamples/sec: ' + str(samples/etime) + '\tApprox. Speed: ' + str(samples * 5 / etime) + 'x real-time'
        print message, '\n'
        f.write(message.replace('\t', s5) + '<br />')
    else:
        print time.strftime('%H:%M:%S') + '\n========\nHIT NAN=======\nInvalidating Epoch\n===========\n'
        f.write('\n' + time.strftime('%H:%M:%S') + '\n========\nHIT NAN=======\nInvalidating Epoch\n===========\n\n')


alphabet = np.arange(29) #[a,....,z, space, ', -]


# Load samples
import os
samples = []
directory = 'pickled'
files = [os.path.join(directory, x) for x in os.listdir(directory)]
files = files[:10]

bad_characters = ['\\', '&', '.', ',', ':', '"', ';', '!']
for f in files:
    submission = pickle.load(open(f, 'rb'))
    print f, '\t\t\t',
    count = 0
    for sample in submission:
        
        label_len = len(sample[0])
        label_prime_len = len(make_l_prime(str_to_seq(F(sample[0], len(alphabet))), len(alphabet)))
        num_buckets = np.shape(sample[1])[0]
        if label_len < 50 and label_prime_len <= num_buckets:# and (float(num_buckets) / float(label_prime_len) < 3.0):
            window_size = 1 # 1 frame of context on each side
            windowed = []
            for i in np.arange(window_size, len(sample[1])-window_size):
                windowed.append(np.concatenate((sample[1][i-1], sample[1][i], sample[1][i+1])))
            
            # Remove bad characters, replace ordinals, create sample
            prompt = sample[0].translate(None, ''.join(bad_characters)).replace('0', 'zero').replace('1', 'one').replace('2', 'two').replace('3', 'three').replace('4', 'four').replace('5', 'five')
            samples.append( (make_l_prime(str_to_seq(F(prompt, len(alphabet))), len(alphabet)), windowed) )
            count += 1
    print count, 'samples selected'
            

visual_sample = samples[0]
visual_sample2 = samples[1]
print 'Shape of first input:', np.shape(visual_sample[1])
print seq_to_str(visual_sample[0])
print seq_to_str(visual_sample2[0])
    


net = nn.Network()
rng = np.random.RandomState(int(time.time()))

try:
    last_good = 18
    restart = False
    
    duration = time.time()
    picklePath = 'cpu5/' + str(last_good) + '.pkl'
    print 'loading from', picklePath
    network = net.load_network(picklePath, 240, len(alphabet)+1, .001, .5)        #x3 for the window
    print 'loaded Network - num samples:', len(samples), '\tDuration: %f' % (time.time()-duration)
    
    epoch_size = 1000 # Samples per epoch
    if len(samples) < epoch_size: epoch_size = len(samples)
    
    for epoch in np.arange(last_good+1, 100000):
        rng.shuffle(samples)
        avg_error = 0.0
        duration = time.time()

        for i in np.arange(epoch_size):
        
            cst, pred = network.trainer(samples[i][1], samples[i][0])
            avg_error += cst

            if math.isnan(avg_error) or math.isinf(avg_error):
                restart = True
                break
        
        log_file = file('/var/www/html/status.html', 'a')
        if not restart:
            log_it(log_file, epoch, error=avg_error, etime=(time.time()-duration), samples=epoch_size)

            if epoch % 1 == 0:
                dumpPath = 'cpu5/' + str(epoch) + '.pkl'
                print 'Saving to: ', dumpPath
                net.dump_network(dumpPath)
                
                pred = network.tester(visual_sample[1])[0]
                print seq_to_str(visual_sample[0]), '||', seq_to_str([np.argmax(x) for x in pred])
                pred = network.tester(visual_sample2[1])[0]
                print seq_to_str(visual_sample2[0]), '||', seq_to_str([np.argmax(x) for x in pred])
                
                last_good = epoch
                
        else:
            log_it(log_file, nan=True)
            restart = False
            
            duration = time.time()
            picklePath = 'cpu5/' + str(last_good) + '.pkl'
            print 'loading from', picklePath
            network = net.load_network(picklePath, 240, len(alphabet)+1, .001, .5)        #x3 for the window
            print 'loaded Network - num samples:', len(samples), '\tDuration: %f' % (time.time()-duration)
        log_file.close()
           
except KeyboardInterrupt:
    action = raw_input('\nDisplay results of all samples? <y,n> ')
    
    if action == 'y':
        for s in samples:
            print seq_to_str(s[0]), '||', seq_to_str([np.argmax(x) for x in network.tester(s[1])[0]])
            
            
