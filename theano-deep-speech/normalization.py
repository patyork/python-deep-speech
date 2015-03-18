__author__ = 'pat'
import os
import cPickle as pickle
import numpy as np
from itertools import groupby

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
directory = 'pickled'
files = [os.path.join(directory, x) for x in os.listdir(directory)]
files = files[:10]


# 'Normalize' the data
# This is not a true normalization, but is actually the normalization with respect to each audio file
#   i.e (The norm of ( the norm of the audio files ) )
# This disregards the length of the audio file in the normalization of the data.
#   Perhaps, this is a worse or better approach, but it is the simplest
means = []
stds = []
samples_x = []
samples_y = []

bad_characters = ['\\', '&', '.', ',', ':', '"', ';', '!']
for f in files:
    submission = pickle.load(open(f, 'rb'))
    print f, '\t\t\t',
    count = 0
    for sample in submission:

        label_len = len(sample[0])
        label_prime_len = len(make_l_prime(str_to_seq(F(sample[0], len(alphabet))), len(alphabet)))
        num_buckets = np.shape(sample[1])[0]
        if label_len < 35 and label_prime_len <= num_buckets:# and (float(num_buckets) / float(label_prime_len) < 3.0):
            ''''
            windowed = []
            for i in np.arange(window_size, len(sample[1])-window_size):
                windowed.append(np.concatenate(([sample[1][i+j] for j in np.arange(-window_size, window_size+1)])))'''''

            # Remove bad characters, replace ordinals, create sample
            prompt = sample[0].translate(None, ''.join(bad_characters)).replace('0', 'zero').replace('1', 'one').replace('2', 'two').replace('3', 'three').replace('4', 'four').replace('5', 'five')
            samples_y.append(make_l_prime(str_to_seq(F(prompt, len(alphabet))), len(alphabet)))

            samples_x.append(sample[1])
            means.append(np.mean(sample[1], axis=0))
            stds.append(np.std(sample[1], axis=0))

            count += 1

    print count, 'samples selected'

mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)
norm_samples = [(samples_x[i] - mean)/std for i in np.arange(len(samples_x))]

windowed_norm_samples = []
window_size = 3             # 3 frames of context on each side

for sample in norm_samples:
    windowed = []
    for i in np.arange(window_size, len(sample)-window_size):
        windowed.append(np.concatenate(([sample[i+j] for j in np.arange(-window_size, window_size+1)])))
    windowed_norm_samples.append(windowed)

normalized = [(samples_y[i], windowed_norm_samples[i]) for i in np.arange(len(windowed_norm_samples))]

f = open('win3_l35.pkl', 'wb')
pickle.dump(normalized, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()






