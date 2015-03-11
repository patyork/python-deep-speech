__author__ = 'pat'
# Development of a log-scale dot product function

import numpy as np


def recurrence_relationship(size):
    big_I = np.eye(size+2)
    return np.eye(size) + big_I[2:, 1:-1] + big_I[2:, :-2] * (np.arange(size) % 2)


def logScaleTest():
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


def parallelCTC():
    rng = np.random.RandomState(1234)
    y1 = rng.random_sample((10,10))
    y2 = rng.random_sample((10,10))
    rec1 = recurrence_relationship(10)
    rec2 = recurrence_relationship(10)

    y3 = np.concatenate((y1, y2), axis=0)
    rec3 = np.tile(recurrence_relationship(10), (2, 1))

    y4 = np.concatenate((y1, y2), axis=1)
    rec4 = np.tile(recurrence_relationship(10), (1, 2))


    # Actual, double loop
    start = np.eye(10)[0]
    prob1 = y1[0] * np.dot(start, rec1[0])
    prob2 = y2[0] * np.dot(start, rec2[0])

    for i in np.arange(1, 10):
        prob1 = y1[i] * np.dot(prob1, rec1[i])
        prob2 = y2[i] * np.dot(prob2, rec2[i])

    #print prob1, np.sum(prob1[-2:])
    #print prob2, np.sum(prob2[-2:])
    print 'Mean:', (np.sum(prob1[-2:]) + np.sum(prob2[-2:]))/2

    print
    print


    # Dot double
    start = np.vstack((np.eye(10)[0], np.eye(10)[0]))
    prob = np.vstack((y1[0], y2[0]))
    print prob
    print '--------------'
    prob = (prob.T * np.dot(start, rec1[0])).T
    for i in np.arange(1,10):
        print prob, '\n\n'
        prob = (np.vstack((y1[i], y2[i])).T * np.dot(prob, rec1[i])).T

    print prob
    print np.sum(prob[:, -2:])/2



    raw_input()


    # tiled horizontally
    start = np.tile(np.eye(4)[0], (1, 2))
    prob = y4[0] * np.dot(start, rec4[0])/2
    for i in np.arange(1, 4):
        prob = y4[i] * np.dot(prob, rec4[i])/2

    #print prob

    #print prob[2] + prob[3]
    #print prob[-2] + prob[-1]
    print ((prob[-2] + prob[-1])+(prob[2] + prob[3]))/2



    print
    print

    # Averaged probabilities
    y5 = (y1+y2)/2
    start = np.eye(4)[0]
    rec5 = recurrence_relationship(4)
    prob = y5[0] * np.dot(start, rec5[0])
    for i in np.arange(1, 4):
        prob = y5[i] * np.dot(prob, rec5[i])

    print np.sum(prob[-2:])

    raw_input()

    print np.dot(y1[0], rec1[0])
    print np.dot(y2[0], rec2[0])

    print
    for i in np.arange(8):
        print np.dot(y3[i], rec3[i])

    print
    print np.dot(y4[0], rec4[0]), np.dot(y1[0], rec1[0])+np.dot(y2[0], rec2[0])


parallelCTC()
