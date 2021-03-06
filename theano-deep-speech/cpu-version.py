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

            

visual_sample = samples[0]
visual_sample2 = samples[1]
print 'Shape of first input:', np.shape(visual_sample[1])
print seq_to_str(visual_sample[0])
print seq_to_str(visual_sample2[0])
    


net = nn.Network()
rng = np.random.RandomState(int(time.time()))

try:
    last_good = 609
    restart = False
    log_file = file('/var/www/html/status.html', 'a')
    duration = time.time()
    if last_good == -1:
        network = net.create_network(560, len(alphabet)+1, .0005, .75)        #x3 for the window
        log_it(log_file, etc='created Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
    else:
        picklePath = 'saved_models/' + str(last_good) + '.pkl'
        print 'loading from', picklePath
        network = net.load_network(picklePath, 560, len(alphabet)+1, .0005, .75)        #x3 for the window
        log_it(log_file, etc='loaded Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
    log_file.close()
    
    epoch_size = 500 # Samples per epoch
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
                dumpPath = 'saved_models/' + str(epoch) + '.pkl'
                print 'Saving to: ', dumpPath
                net.dump_network(dumpPath)
                
                pred = network.tester(visual_sample[1])[0]
                log_it(log_file, etc='\t' + seq_to_str(visual_sample[0]) + ' || ' + seq_to_str([np.argmax(x) for x in pred]))
                pred2 = network.tester(visual_sample2[1])[0]
                log_it(log_file, etc='\t' + seq_to_str(visual_sample2[0]) + ' || ' + seq_to_str([np.argmax(x) for x in pred2]) + '\t===Diff: ' + str(np.mean(pred2-pred)))
                
                last_good = epoch
                
        else:
            log_it(log_file, nan=True)
            restart = False
            
            duration = time.time()
            picklePath = 'saved_models/' + str(last_good) + '.pkl'
            print 'loading from', picklePath
            network = net.load_network(picklePath, 560, len(alphabet)+1, .0005, .75)        #x3 for the window
            log_it(log_file, etc='loaded Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
        log_file.close()
           
except KeyboardInterrupt:
    action = raw_input('\nDisplay results of all samples? <y,n> ')
    
    if action == 'y':
        for s in samples:
            print seq_to_str(s[0]), '||', seq_to_str([np.argmax(x) for x in network.tester(s[1])[0]])
       
            
            
