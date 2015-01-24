__author__ = 'pat'
import numpy as np
import math
import time

def softmax(x):
    e = np.exp(x)
    return e / np.sum(np.exp(x))

def logistic(x): return 1.0/(1.0 + np.exp(-x))


def dlogistic(y):
    logi = logistic(y)
    return logi * (1.0 - logi)


def softplus(x):
    return min(max(0, x), 20)
    #y = math.log(1.0 + np.exp(x))
    #if y > 20: return 20
    #return y


def dsoftplus(y):
    return logistic(y)


def linear(x):
    return x


def dlinear(_):
    return 1.0


def tanh(x):
    return math.tanh(x)


def dtanh(y):
    return 1.0 - y**2


def gaussian(x):
    return np.exp(-1.0 * x**2 / 2)


def dgaussian(y):
    return gaussian(y) * -y


class FeedForwardLayer:
    def __init__(self, rng, nin, nout, activation=logistic, activation_prime=dlogistic, W=None, b=None, inputs=None, learningRate=.9):
        self.ActivationFn = np.frompyfunc(activation, 1, 1)
        self.DActivationFn = np.frompyfunc(activation_prime, 1, 1)
        self.inputs = inputs
        self.outputs = None
        self.activations = None
        self.learningRate = learningRate
        self.momentumFactor = .7
        self.previousDelta = None
        self.previousbDelta = None

        if not W:
            self.W = np.asarray(
                rng.uniform(
                    low=4 * np.sqrt(6.0 / (nin + nout)),     # generic range of values
                    high=-4 * np.sqrt(6.0 / (nin + nout)),
                    size=(nin, nout)
                )
                , dtype=float
            )
        else:
            self.W = W
        if not b:
            self.b = np.zeros(nout)
        else:
            self.b = b

    def predict(self, inputs):
        self.inputs = inputs
        self.activations = np.dot(self.inputs, self.W) + self.b
        self.outputs = np.asarray(self.ActivationFn(self.activations), dtype=float)
        return self.outputs

    def predict_series(self, series):
        out = []
        for window in series:
            out.append(np.asarray(self.predict(window), dtype=float))
        return out

    def update(self, targets=None, sensitivityConstants=None):

        if targets is not None:
            sens = np.subtract(self.outputs, targets) * self.DActivationFn(self.activations)

            # print 'Sensitivities:', sens
            # print 'Inputs:', self.input
            # print 'Tiled inputs', np.tile(self.input, (len(sens), 1))

            d = []
            for s in sens:
                for i in self.inputs:
                    d.append(s * i)
            gradient = np.array(d).reshape(self.W.shape)
            # gradient = np.multiply(sens, np.tile(self.input, (len(sens), 1)))



            delta = (-self.learningRate * gradient).reshape(self.W.shape)
            bdelta = (-self.learningRate * sens).reshape(self.b.shape)
            self.W += delta + (self.momentumFactor * self.previousDelta if self.previousDelta is not None else 0)
            self.b += bdelta + (self.momentumFactor * self.previousbDelta if self.previousbDelta is not None else 0)
            self.previousDelta = delta
            self.previousbDelta = bdelta

            sensConstants = np.multiply(sens, self.W)
            return np.sum(sensConstants, axis=1)

        elif sensitivityConstants is not None:
            sens = np.multiply(sensitivityConstants.transpose(), self.DActivationFn(self.activations)).transpose()

            d=[]
            for s in sens:
                for i in self.inputs:
                    d.append(s*i)
            gradient = np.array(d).reshape(self.W.shape)
            #gradient = sens * self.input


            delta = (-self.learningRate * gradient).reshape(self.W.shape)
            bdelta = (-self.learningRate * sens).reshape(self.b.shape)
            self.W += delta + (self.momentumFactor * self.previousDelta if self.previousDelta is not None else 0)
            self.b += bdelta + (self.momentumFactor * self.previousbDelta if self.previousbDelta is not None else 0)
            self.previousDelta = delta
            self.previousbDelta = bdelta

            sensConstants = np.multiply(sens, self.W)
            return np.sum(sensConstants, axis=1)


class RecurrentForwardLayer(FeedForwardLayer):
    def update(self):
        pass


class RecurrentBackwardLayer(FeedForwardLayer):
    def update(self):
        pass


# This class acts as a wrapper for a layer that consists of a Forward and a Backward recurrent layer
class BiDirectionalLayer:
    def __init__(self, forward_layer, backward_layer, initial_state):
        self.forward = forward_layer
        self.backward = backward_layer
        self.forward.outputs = initial_state
        self.backward.outputs = initial_state

    def predict_series(self, windows):
        # Remember to reverse the list of windows for the Backwards layer!

        bd_output = np.concatenate(
            (
                # Forward pass over inputs and F(t-1)
                [self.forward.predict(np.concatenate((win, self.forward.outputs), axis=0)) for win in windows],
                # Backward pass over inputs and B(t+1)
                [self.backward.predict(np.concatenate((win, self.backward.outputs), axis=0)) for win in windows[::-1]]
            ),
            axis=1
        )
        return bd_output


class BDRNN:
    def __init__(self, defined_layers=None, window_width=0):
        self.layers = defined_layers
        assert(self.layers is not None)

        self.window_width = window_width

    def apply_window(self, input_stream):
        windows = []

        for i in range(self.window_width, len(input_stream)-self.window_width):
            windows.append(np.asarray(input_stream[i-self.window_width:i+self.window_width+1], dtype=float).flatten())

        return windows

    def predict(self, input_stream):
        assert(self.layers is not None)

        # get the output from the first layer for each window of data
        '''if isinstance(self.layers[0], RecurrentForwardLayer) or isinstance(self.layers[0], RecurrentBackwardLayer):
            outputs = self.layers[0].predict(input_stream)
            print 'hit bd'
        else:
            outputs = [np.asarray(self.layers[0].predict(win), dtype=float) for win in self.apply_window(input_stream)]
            print 'hit ff' '''
        outputs = self.layers[0].predict_series(self.apply_window(input_stream))

        # repeat for following layers
        for i in range(1, len(self.layers)):
            '''if isinstance(self.layers[i], RecurrentForwardLayer) or isinstance(self.layers[i], RecurrentBackwardLayer):
                outputs = self.layers[i].predict(outputs)
                print 'hit bd'
            else:
                outputs = [np.asarray(self.layers[i].predict(win), dtype=float) for win in outputs]
                print 'hit ff' '''
            outputs = self.layers[i].predict_series(outputs)

        return outputs


    def train(self, input_streams):
        for input_stream in input_streams:
            pass

def test():
    test_2d_sequence = np.asarray([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ])
    print type(test_2d_sequence)

    win_size = 0
    dimensionality = 2
    rng = np.random.RandomState(123)

    h1 = FeedForwardLayer(rng, 2*dimensionality*win_size + dimensionality, 3)
    h2 = FeedForwardLayer(rng, 3, 3)
    h3 = FeedForwardLayer(rng, 3, 2)

    recurrency_output_count = 2
    f1 = RecurrentForwardLayer(rng, 2 + recurrency_output_count, recurrency_output_count)
    b1 = RecurrentBackwardLayer(rng, 2 + recurrency_output_count, recurrency_output_count)
    bd1 = BiDirectionalLayer(f1, b1, initial_state=np.zeros(recurrency_output_count))

    o = FeedForwardLayer(rng, 2*recurrency_output_count, 2)

    bdrnn = BDRNN([h1, h2, h3, bd1, o], window_width=win_size)



    print bdrnn.predict(test_2d_sequence)

    t0 = time.time()
    for i in range(10000):
        bdrnn.predict(test_2d_sequence)
    print time.time() - t0