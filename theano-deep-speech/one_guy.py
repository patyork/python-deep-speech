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
files = files[:10]

if DEBUG:
    #files = files[:30]
    pass





for f in files:
    submission = pickle.load(open(f, 'rb'))
    print f

    for sample in submission:
        label_len = len(sample[0])
        label_prime_len = len(make_l_prime(str_to_seq(F(sample[0], len(alphabet))), len(alphabet)))
        num_buckets = np.shape(sample[1])[0]
        if 10 < label_len < 45 and label_prime_len <= num_buckets:# and (float(num_buckets) / float(label_prime_len) < 3.0):
            samples.append(sample)

#samples = samples[:1]

network = nn.BRNN(np.shape(samples[0][1])[1]*3, len(alphabet)+1)        #x3 for the window

print 'built network - num samples:', len(samples)

# Remove smaples that cause inf right off of the bat
bad = []
indexes = 0
for s in samples:
    label_prime = make_l_prime(str_to_seq(F(s[0], len(alphabet))), len(alphabet))

    # Window and send
    window_size = 1 # 1 frame of context on each side
    windowed = []
    for i in np.arange(window_size, len(s[1])-window_size):
        windowed.append(np.concatenate((s[1][i-1], s[1][i], s[1][i+1])))

    windowed = np.asarray(windowed)

    cst = network.validator(windowed, label_prime)[0]

    if math.isinf(cst) or math.isnan(cst):
        bad.append(indexes)

    indexes += 1

for ndx in bad[::-1]:
    del samples[ndx]

print 'Samples after inf test:', len(samples)

if TEST:
    samples = samples[:1]

if DEBUG:
    for s in samples:
        label_prime = make_l_prime(str_to_seq(F(s[0], len(alphabet))), len(alphabet))

        # Window and send
        window_size = 1 # 1 frame of context on each side
        windowed = []
        for i in np.arange(window_size, len(s[1])-window_size):
            windowed.append(np.concatenate((s[1][i-1], s[1][i], s[1][i+1])))

        windowed = np.asarray(windowed)


        if not len(label_prime) > np.shape(s[1])[0]:
            cst, pred = network.trainer(windowed, label_prime)

            if math.isinf(cst) or math.isnan(cst):
                print 'HEY!', np.shape(s[1]), len(s[0]), len(label_prime), cst, s[0]
                #print str(np.shape(s[1])[0]) + ',' + str(len(label_prime))


    raw_input('end program')


visual_sample = samples[0]
plt.ion()
figure = plt.figure()
sp = figure.add_subplot(1,1,1)
line0 = Line2D([], [], color='#FF9300', linewidth=1)
line1 = Line2D([], [], color='#E84B0C', linewidth=1)
line2 = Line2D([], [], color='#FF0000', linewidth=1)
line3 = Line2D([], [], color='#E80CE5', linewidth=1)
line4 = Line2D([], [], color='#7F0DFF', linewidth=1)
line5 = Line2D([], [], color='#00C6FF', linewidth=1)
line6 = Line2D([], [], color='#0CE8A9', linewidth=1)
line7 = Line2D([], [], color='#00FF2F', linewidth=1)
line8 = Line2D([], [], color='#6AE80C', linewidth=1)
line9 = Line2D([], [], color='#FFF900', linewidth=1)
line10 = Line2D([], [], color='#0029FF', linewidth=1)
line11 = Line2D([], [], color='#580CE8', linewidth=1)
line12 = Line2D([], [], color='#FFA140', linewidth=1)
line13 = Line2D([], [], color='#E85B2F', linewidth=1)
line14 = Line2D([], [], color='#FF4058', linewidth=1)
line15 = Line2D([], [], color='#E22FE8', linewidth=1)
line16 = Line2D([], [], color='#9740FF', linewidth=1)
line17 = Line2D([], [], color='#4CFFE0', linewidth=1)
line18 = Line2D([], [], color='#39E86D', linewidth=1)
line19 = Line2D([], [], color='#7FFF4C', linewidth=1)
line20 = Line2D([], [], color='#DDE839', linewidth=1)
line21 = Line2D([], [], color='#FFE04C', linewidth=1)
line22 = Line2D([], [], color='#6174FF', linewidth=1)
line23 = Line2D([], [], color='#884CE8', linewidth=1)
line24 = Line2D([], [], color='#FFA5F6', linewidth=1)
line25 = Line2D([], [], color='#F6EDFF', linewidth=1)
line26 = Line2D([], [], color='#A9A5FF', linewidth=1)
line27 = Line2D([], [], color='#8BB6E8', linewidth=1)
line28 = Line2D([], [], color='#A5FBFF', linewidth=1)
line29 = Line2D([], [], color='black', linewidth=2)
sp.add_line(line0)
sp.add_line(line1)
sp.add_line(line2)
sp.add_line(line3)
sp.add_line(line4)
sp.add_line(line5)
sp.add_line(line6)
sp.add_line(line7)
sp.add_line(line8)
sp.add_line(line9)
sp.add_line(line10)
sp.add_line(line11)
sp.add_line(line12)
sp.add_line(line13)
sp.add_line(line14)
sp.add_line(line15)
sp.add_line(line16)
sp.add_line(line17)
sp.add_line(line18)
sp.add_line(line19)
sp.add_line(line20)
sp.add_line(line21)
sp.add_line(line22)
sp.add_line(line23)
sp.add_line(line24)
sp.add_line(line25)
sp.add_line(line26)
sp.add_line(line27)
sp.add_line(line28)
sp.add_line(line29)
sp.set_ylim(-.1, 1.1)
sp.set_xlim(0, len(visual_sample[1])-1)

print 'Shape of first input:', np.shape(visual_sample[1])
print visual_sample[0]

rng = np.random.RandomState(1234)

try:
    for epoch in np.arange(100000):
        rng.shuffle(samples)

        avg_error = 0.0

        duration = time.time()

        for s in samples:
            label_prime = make_l_prime(str_to_seq(F(s[0], len(alphabet))), len(alphabet))

            # Window and send
            window_size = 1 # 1 frame of context on each side
            windowed = []
            for i in np.arange(window_size, len(s[1])-window_size):
                windowed.append(np.concatenate((s[1][i-1], s[1][i], s[1][i+1])))

            windowed = np.asarray(windowed)

            cst, pred = network.trainer(windowed, label_prime)
            avg_error += cst

        if math.isnan(avg_error) or math.isinf(avg_error):

            for x in np.arange(10):
                [' ' for y in range(x)]
                print [' ' for y in range(x)], 'Hit nan/inf'

                raw_input('hit NAN')

        print 'Epoch:', epoch, '\t', avg_error / len(samples), str(time.time()-duration)+'s', '\t' + str(len(samples)), 'samples'

        if epoch % 1 == 0:
            # Window and send
            window_size = 1 # 1 frame of context on each side
            windowed = []
            for i in np.arange(window_size, len(visual_sample[1])-window_size):
                windowed.append(np.concatenate((visual_sample[1][i-1], visual_sample[1][i], visual_sample[1][i+1])))

            windowed = np.asarray(windowed)

            pred = network.tester(windowed)[0]


            xdata = np.arange(len(pred))
            #print len(pred), [x[29] for x in pred]
            line0.set_data(xdata, [x[0] for x in pred])
            line1.set_data(xdata, [x[1] for x in pred])
            line2.set_data(xdata, [x[2] for x in pred])
            line3.set_data(xdata, [x[3] for x in pred])
            line4.set_data(xdata, [x[4] for x in pred])
            line5.set_data(xdata, [x[5] for x in pred])
            line6.set_data(xdata, [x[6] for x in pred])
            line7.set_data(xdata, [x[7] for x in pred])
            line8.set_data(xdata, [x[8] for x in pred])
            line9.set_data(xdata, [x[9] for x in pred])
            line10.set_data(xdata, [x[10] for x in pred])
            line11.set_data(xdata, [x[11] for x in pred])
            line12.set_data(xdata, [x[12] for x in pred])
            line13.set_data(xdata, [x[13] for x in pred])
            line14.set_data(xdata, [x[14] for x in pred])
            line15.set_data(xdata, [x[15] for x in pred])
            line16.set_data(xdata, [x[16] for x in pred])
            line17.set_data(xdata, [x[17] for x in pred])
            line18.set_data(xdata, [x[18] for x in pred])
            line19.set_data(xdata, [x[19] for x in pred])
            line20.set_data(xdata, [x[20] for x in pred])
            line21.set_data(xdata, [x[21] for x in pred])
            line22.set_data(xdata, [x[22] for x in pred])
            line23.set_data(xdata, [x[23] for x in pred])
            line24.set_data(xdata, [x[24] for x in pred])
            line25.set_data(xdata, [x[25] for x in pred])
            line26.set_data(xdata, [x[26] for x in pred])
            line27.set_data(xdata, [x[27] for x in pred])
            line28.set_data(xdata, [x[28] for x in pred])
            line29.set_data(xdata, [x[29] for x in pred])

            figure.canvas.draw()
            plt.pause(0.01)
except KeyboardInterrupt:
    for s in samples:
        # Window and send
        window_size = 1 # 1 frame of context on each side
        windowed = []
        for i in np.arange(window_size, len(s[1])-window_size):
            windowed.append(np.concatenate((s[1][i-1], s[1][i], s[1][i+1])))

        windowed = np.asarray(windowed)

        print s[0], '||', seq_to_str([np.argmax(x) for x in network.tester(windowed)[0]])

raw_input('asdgf')