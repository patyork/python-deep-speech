__author__ = 'pat'
# Development of a log-scale dot product function

import numpy as np


def recurrence_relationship(size):
    big_I = np.eye(size+2)
    return np.eye(size) + big_I[2:, 1:-1] + big_I[2:, :-2] * (np.arange(size) % 2)

rng = np.random.RandomState(1234)

vec = rng.randint(0, 10, 5)*2.5e-47
rec = recurrence_relationship(5)
rec = np.ones((5,5), dtype='float64')

ACTUAL = np.log(np.dot(vec.astype('float64'), rec.astype('float64')))
#print ACTUAL

print np.dot(np.log(vec.astype('float32')), rec.astype('float32'))/5+1.718

def log_dot(vector, matrix):
    print 'mat', matrix.dtype
    print 'vec', vector.dtype
    vector = vector.astype('float32')
    matrix = matrix.astype('float32')
    print 'mat', matrix.dtype
    print 'vec', vector.dtype
    out = []
    for row in matrix:
        running = 0
        for i in np.arange(len(row)):
            b = row[i]*vector[i]
            if not b==0:
                lnb = np.log(b)
                running += np.log(np.exp(lnb - running))
        out.append(running)
    return out

logvers = np.asarray(log_dot(vec, rec), dtype='float32')

print ACTUAL
print logvers
print logvers-ACTUAL
print 'logvers', logvers.dtype
print 'actual', ACTUAL.dtype
