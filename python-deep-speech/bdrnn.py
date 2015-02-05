__author__ = 'pat'
import numpy as np
import math
import time
from itertools import groupby


def softmax(x):
    e = np.exp(x)
    return e / np.sum(np.exp(x))


def logistic(x):
    #"Numerically-stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)
    #return 1.0/(1.0 + np.exp(-x))


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
        self.activations = None
        self.activationHistory = None
        self.outputs = None
        self.outputHistory = None
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
            return np.average(sensConstants, axis=1)


class FeedForwardLayerRNN:
    def __init__(self, rng, nin, nout, activation=logistic, activation_prime=dlogistic, W=None, b=None, inputs=None, learningRate=.2):
        self.ActivationFn = np.frompyfunc(activation, 1, 1)
        self.DActivationFn = np.frompyfunc(activation_prime, 1, 1)

        self.inputs = []
        if inputs: self.inputs.append(inputs)
        self.activations = []
        self.outputs = []

        self.learningRate = learningRate
        self.momentumFactor = 0     # TODO: ensure momentum is still implemented correctly
        self.previousDelta = None
        self.previousbDelta = None

        #W = np.zeros((nin, nout))
        if W is None:
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
        self.inputs.append(inputs)
        self.activations.append(np.dot(inputs, self.W) + self.b)
        self.outputs.append(np.asarray(self.ActivationFn(self.activations[-1]), dtype=float))
        return self.outputs[-1]

    def predict_series(self, series):
        # TODO: this can be optimized, probably

        self.activations = []
        self.inputs = []
        self.outputs = []

        out = []
        for window in series:
            out.append(np.asarray(self.predict(window), dtype=float))

        #return self.outputs
        return out

    def update(self, targets=None, sensitivityConstants=None):

        # If this is an output layer with a well defined notion of a target series
        if targets is not None:
            # TODO: implement and test this functionality for cases where CTC is not used
            for target in targets:
                sens = np.subtract(self.outputs, targets) * self.DActivationFn(self.activations)

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
            #print len(self.activations), len(sensitivityConstants), np.shape(self.activations), np.shape(sensitivityConstants)
            # Ensure that we have as many sensitivity constants as we do timesteps/activations
            assert(len(self.activations)==len(sensitivityConstants))

            # Find the new deltas using what we know from the layer above (SUM(deltas*weights) == sensitivity constants
            backwardSensConstants = []
            wDeltas = []
            bDeltas = []
            for t in np.arange(len(self.activations)):
                sens = np.multiply(np.asarray(sensitivityConstants[t]).transpose(), self.DActivationFn(self.activations[t])).transpose()

                d=[]
                for s in sens:
                    for i in self.inputs[t]:
                        d.append(s*i)
                gradient = np.array(d).reshape(self.W.shape)
                #gradient = sens * self.input

                # Calculate sensConstants before updating weights
                sensConstants = np.multiply(sens, self.W)
                backwardSensConstants.append(np.sum(sensConstants, axis=1))

                wDeltas.append((-self.learningRate * gradient).reshape(self.W.shape))
                bDeltas.append((-self.learningRate * sens).reshape(self.b.shape))


                #delta = (-self.learningRate * gradient).reshape(self.W.shape)
                #bdelta = (-self.learningRate * sens).reshape(self.b.shape)
                #self.W += delta + (self.momentumFactor * self.previousDelta if self.previousDelta is not None else 0)
                #self.b += bdelta + (self.momentumFactor * self.previousbDelta if self.previousbDelta is not None else 0)
                #self.previousDelta = delta
                #self.previousbDelta = bdelta

            # Update batch weights
            self.W += np.sum(wDeltas, axis=0)
            self.b += np.sum(bDeltas, axis=0)
            return backwardSensConstants


class RecurrentForwardLayer:
    def __init__(self, rng, nin, nout, activation=logistic, activation_prime=dlogistic, W=None, b=None, inputs=None, learningRate=.2):
        self.ActivationFn = np.frompyfunc(activation, 1, 1)
        self.DActivationFn = np.frompyfunc(activation_prime, 1, 1)
        self.inputs = []
        if inputs:
            self.inputs.append(inputs)
        self.activations = []
        self.outputs = []
        self.learningRate = learningRate
        self.momentumFactor = .7
        self.previousDelta = None
        self.previousbDelta = None

        self.nout = nout
        self.nin = nin

        # This is the number of inputs that come from the layer below;
        # ..excludes the number of inputs from previous timesteps
        self.prev_layer_in = nin-nout

        #W = np.zeros((nin, nout))
        if W is None:
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
        _inputs = inputs
        _activations = np.dot(_inputs, self.W) + self.b
        _outputs = np.asarray(self.ActivationFn(_activations), dtype=float)

        # We have to keep track of inputs/activations/outputs for each time/sequence step
        self.inputs.append(_inputs)
        self.activations.append(_activations)
        self.outputs.append(_outputs)
       # list(self.outputs).append(_outputs)

        return _outputs

    def predict_series(self, series):
        self.inputs = []
        self.activations = []
        self.outputs = []

        out = []
        for window in series:
            out.append(np.asarray(self.predict(window), dtype=float))
        return out

    def update(self, sensitivityConstants):
        assert(len(self.activations)==len(sensitivityConstants))

        # The update for the forward layer must go back in time (T-1 -> 0)
        # We must also loop over the activations/inputs in backwards order
        backwardSensConstants = []
        wDeltas = []
        bDeltas = []

        for t in np.arange(len(self.activations)-1, -1, -1):
            # Rationalization: the sensitivity constants are the dot product for all weights coming into the output layer
            # That means that only some of the sensitivity constants are applicable to the forward or backward layer
            # ...namely, the first half.

            # We don't have any sensitivity constants from this layer for a time step that is outside of the bounds
            # So, we let the first set of sensitivity constants be the zero vector
            # (or simply don't add anything to the sensitivity constants from the output layer)
            if t==len(self.activations)-1:
                sens = np.multiply(np.asarray(sensitivityConstants[t][:self.nout]).transpose(), self.DActivationFn(self.activations[t])).transpose()
            else:
                temp1 = np.asarray(sensitivityConstants[t][:self.nout]) + backwardSensConstants[-1][self.prev_layer_in:]
                sens = np.multiply(np.asarray(temp1).transpose(), self.DActivationFn(self.activations[t])).transpose()


            d=[]
            for s in sens:
                for i in self.inputs[t]:
                    d.append(s*i)
            gradient = np.array(d).reshape(self.W.shape)
            #gradient = sens * self.input

            # Calculate sensConstants before updating weights
            sensConstants = np.multiply(sens, self.W)
            backwardSensConstants.append(np.sum(sensConstants, axis=1))

            wDeltas.append((-self.learningRate * gradient).reshape(self.W.shape))
            bDeltas.append((-self.learningRate * sens).reshape(self.b.shape))

        # update weights
        self.W += np.sum(wDeltas, axis=0)
        self.b += np.sum(bDeltas, axis=0)
        return backwardSensConstants[::-1]


class RecurrentBackwardLayer(RecurrentForwardLayer):
    def update(self, sensitivityConstants):
        assert(len(self.activations)==len(sensitivityConstants))

        backwardSensConstants = []
        wDeltas = []
        bDeltas = []

        # Activations are in order or t=T-1....0
        # We must loop over them backwards
        # the Sensitivity Constants from the output layer are in order t=0....T-1, but are reversed before this call
        for t in np.arange(len(self.activations)-1, -1, -1):

            # See RecurrentForwardLayer for an explanation of the slicing of the sensitivityConstants
            if t==len(self.activations)-1:
                sens = np.multiply(np.asarray(sensitivityConstants[t][self.nout:]).transpose(), self.DActivationFn(self.activations[t])).transpose()
            else:
                temp1 = np.asarray(sensitivityConstants[t][self.nout:]) + backwardSensConstants[-1][self.prev_layer_in:]
                sens = np.multiply(np.asarray(temp1).transpose(), self.DActivationFn(self.activations[t])).transpose()

            d=[]
            for s in sens:
                for i in self.inputs[t]:
                    d.append(s*i)
            gradient = np.array(d).reshape(self.W.shape)
            #gradient = sens * self.inputs

            # Calculate sensConstants before updating weights
            sensConstants = np.multiply(sens, self.W)
            backwardSensConstants.append(np.sum(sensConstants, axis=1))

            wDeltas.append((-self.learningRate * gradient).reshape(self.W.shape))
            bDeltas.append((-self.learningRate * sens).reshape(self.b.shape))

        # update batch weights
        self.W += np.sum(wDeltas, axis=0)
        self.b += np.sum(bDeltas, axis=0)
        return backwardSensConstants


# This class acts as a wrapper for a layer that consists of a Forward and a Backward recurrent layer
class BiDirectionalLayer:
    def __init__(self, forward_layer, backward_layer, initial_state):
        self.forward = forward_layer
        self.backward = backward_layer

        self.initial_state = initial_state
        self.forward.outputs = initial_state
        self.backward.outputs = initial_state

    def predict_series(self, windows):
        # New series; clear the histories
        self.forward.inputs = []
        self.forward.activations = []
        self.forward.outputs = []
        self.forward.outputs.append(self.initial_state)

        self.backward.inputs = []
        self.backward.activations = []
        self.backward.outputs = []
        self.backward.outputs.append(self.initial_state)

        # Remember to reverse the list of windows for the Backwards layer!
        bd_output = np.concatenate(
            (
                # Forward pass over inputs and F(t-1)
                # ...where F(t-1) corresponds to the last self.outputs
                [self.forward.predict(np.concatenate((win, self.forward.outputs[-1]), axis=0)) for win in windows],
                # Backward pass over inputs and B(t+1)
                # ...where B(t+1) corresponds to the last self.outputs
                [self.backward.predict(np.concatenate((win, self.backward.outputs[-1]), axis=0)) for win in windows[::-1]][::-1]
            ),
            axis=1
        )
        return bd_output

    def update(self, sensitivityConstants=None):
        # TODO: retrieve the sentitvity constants from the Bidirectional layer correctly
        fs = self.forward.update(sensitivityConstants)
        bs = self.backward.update(sensitivityConstants[::-1])

        senCat = np.column_stack((np.sum(fs, axis=1), np.sum(bs, axis=1)))

        #print 'out of BDL:', np.shape(senCat)

        return senCat


class SoftmaxLayer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def predict_series(self, windows):
        self.inputs = windows
        #win_max = [(np.max(x), x) for x in windows]
        out = [np.exp(x-np.max(x)) for x in windows]
        out = [x / np.sum(x) for x in out]
        self.outputs = out

        return out

# TODO: Implement a non-recursive CTC algorithm
# TODO: Implement the CTC algorithm in log-scale to prevent underflow; exponentiate only after the summation?
class CTCLayer:
    def __init__(self, alphabet=np.arange(28), remove_duplicates=True):
        self.A = alphabet
        self.blank = len(alphabet)
        self.inputs = None
        self.sequence = None
        self.sequence_prime = None
        self.matrixf = None
        self.matrixb = None

        self.T = None
        self.U = None

        self.remove_duplicates=remove_duplicates

    def recurrence_relationship(self, size):
        big_I = np.eye(size+2)
        return np.eye(size) + big_I[2:, 1:-1] + big_I[2:, :-2] * (np.arange(size) % 2)

    # Remove consecutive symbols and blanks
    def F(self, pi):
        return [a for a in [key for key, _ in groupby(pi)] if a != self.blank]

    # Insert blanks between unique symbols, and at the beginning and end
    def make_l_prime(self, l):
        result = [self.blank] * (len(l) * 2 + 1)
        result[1::2] = l
        return result
        # return [blank] + sum([[i, blank] for i in l], [])

    # Calculate p(sequence|inputs)
    def ctc(self, inputs, sequence):
        self.inputs = inputs
        self.T = len(inputs)
        self.sequence = sequence
        if self.remove_duplicates:
            self.sequence_prime = self.make_l_prime(self.F(sequence))
        else:
            self.sequence_prime = self.make_l_prime(sequence)
        self.U = len(self.sequence_prime)
        self.matrixf = np.zeros((self.T, self.T))
        self.matrixb = np.zeros((self.T, self.T))



        fp = self.forward(self.T-1, self.U-1)
        #print 'matf\n', self.matrixf, 'endmatf'
        self.matrixf = np.zeros((self.T, self.T))
        return fp + self.forward(self.T-1, self.U-2)

        summation = 0.0
        for s in np.arange(self.U):
            self.matrixf = np.zeros((self.T, self.T))
            self.matrixb = np.zeros((self.T, self.T))

            summation += self.forward(self.T-1, s) * self.backward(self.T-1, s) / y[self.T-1][self.sequence_prime[s]]
        return fp

    # DP (recursive) as described by the paper
    def forward(self, t, u):
        if self.matrixf[t][u] != 0:
            return self.matrixf[t][u]

        if t==0 and u==0:
            prob = self.inputs[0][self.blank]
            self.matrixf[t][u] = prob
            return prob
        elif t==0 and u==1:
            prob = self.inputs[0][self.sequence[0]]
            self.matrixf[t][u] = prob
            return prob
        elif t==0: return 0
        elif u<1 or u<(len(self.sequence_prime) - 2*(self.T-1 - t)-1): return 0

        if self.sequence_prime[u]==self.blank or self.sequence_prime[u-2]==self.sequence_prime[u]:
            prob = (self.forward(t-1, u) +
                    self.forward(t-1, u-1)) *\
                    self.inputs[t][self.sequence_prime[u]]
            self.matrixf[t][u] = prob
            return prob
        else:
            prob = (self.forward(t-1, u) +
                    self.forward(t-1, u-1) +
                    self.forward(t-1, u-2)) *\
                    self.inputs[t][self.sequence_prime[u]]
            self.matrixf[t][u] = prob
            return prob

    def backward(self, t, u):
        if self.matrixb[t][u] != 0:
            self.matrixb[t][u]

        if t==self.T-1 and u==len(self.sequence_prime)-1:
            prob = self.inputs[self.T-1][self.blank]
            self.matrixb[t][u] = prob
            return prob
        elif t==self.T-1 and u==len(self.sequence_prime)-2:
            prob = self.inputs[self.T-1][self.sequence[-1]]
            self.matrixb[t][u] = prob
            return prob
        elif t==self.T-1:
            return 0
        elif u>2*t-1 or u>len(self.sequence_prime)-1:
            return 0

        if self.sequence_prime[u]==self.blank:
            prob = (self.backward(t+1, u) +
                    self.backward(t+1, u+1)) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob

        # this is almost certainly incorrect, but there is an out-of-bounds error without it that I cannot track down
        if u==len(self.sequence_prime)-2: return 0

        elif self.sequence_prime[u+2]==self.sequence_prime[u]:
            prob = (self.backward(t+1, u) +
                    self.backward(t+1, u+1)) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob
        else:
            prob = (self.backward(t+1, u) +
                    self.backward(t+1, u+1) +
                    self.backward(t+1, u+2)) *\
                self.inputs[t][self.sequence_prime[u]]
            self.matrixb[t][u] = prob
            return prob

    def alpha_beta(self, inputs, sequence):
        self.inputs = inputs
        self.T = len(inputs)
        self.sequence = sequence
        self.sequence_prime = self.make_l_prime(self.F(sequence))
        self.U = len(self.sequence_prime)
        self.matrixf = np.zeros((self.T, self.T))
        self.matrixb = np.zeros((self.T, self.T))

        alpha_beta = []

        for t in np.arange(self.T):
            ab_t = []
            for u in np.arange(self.U):
                self.matrixf = np.zeros((self.T, self.T))
                self.matrixb = np.zeros((self.T, self.T))
                _f = self.forward(t, u)
                _b = self.backward(t, u)
                ab_t.append(_f * _b)
            alpha_beta.append(ab_t)

        return np.asarray(alpha_beta, dtype=float)


class BDRNN:
    def __init__(self, defined_layers=None, window_width=0, learning_rate=.9, annealing_fn=lambda x: x):
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
        outputs = self.layers[0].predict_series(self.apply_window(input_stream))

        # repeat for following layers
        for i in range(1, len(self.layers)):
            outputs = self.layers[i].predict_series(outputs)

        return outputs

    # Train upon a single input sequence
    def train(self, input_stream, z, ctc_layer):
        y = self.predict(input_stream)      # output from the softmax layer

        aB = ctc_layer.alpha_beta(y, z)
        #p_z_x = ctc_layer.ctc(y, z)                     # TODO: ensure this is a correct approach, or use the summation
        p_z_x = (np.sum(aB[-1]))   #/y[-1])[-1]
        z_prime = ctc_layer.make_l_prime(ctc_layer.F(z))
        #print p_z_x

        # calculate the derivative of the loss with respect to the activations at the output
        K = []
        for k in np.arange(len(ctc_layer.A)+1):
            K.append([x for x in np.where(z_prime==k)[0]])

        sensitivities = []
        for t in np.arange(ctc_layer.T):
            sens_t = []
            for k in np.arange(len(K)):

                summation = 0.0
                for u in K[k]:
                    summation += aB[t][u]

                sens_t.append(y[t][k] - summation/p_z_x)
                #if p_z_x != 0.0:
                #    sens_t.append(y[t][k] - summation/p_z_x)
                #else: sens_t.append(summation)
            sensitivities.append(sens_t)

        #print sensitivities
        # Backpropagate (through time)
        for i in np.arange(len(self.layers)-2, -1, -1):
            sensitivities = self.layers[i].update(sensitivityConstants=sensitivities)   # Output layer

        return p_z_x