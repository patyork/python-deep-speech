__author__ = 'pat'
'''
Bidirectional Recurrent Neural Network
with Connectionist Temporal Classification (CTC)
  courtesy of https://github.com/shawntan/rnn-experiment
  courtesy of https://github.com/rakeshvar/rnn_ctc
implemented in Theano and optimized for use on a GPU
'''

import theano
import theano.tensor as T
from theano_toolkit import utils as U
from theano_toolkit import updates
import numpy as np
import cPickle as pickle

#THEANO_FLAGS='device=cpu,floatX=float32'
#theano.config.warn_float64='warn'

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity='high'


class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        if parameters is None:
            W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            W = theano.shared(parameters['W'], name='W')
            b = theano.shared(parameters['b'], name='b')

        self.output, _ = theano.scan(
            lambda element: self.activation_fn(T.dot(element, W) + b),
            sequences=[inputs]
        )

        self.params = [W, b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params



class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, is_backward=False, parameters=None):

        if parameters is None:
            W_if = U.create_shared(U.initial_weights(input_size, output_size), name='W_if')
            W_ff = U.create_shared(U.initial_weights(output_size, output_size), name='W_ff')
            b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            W_if = theano.shared(parameters['W_if'], name='W_if')
            W_ff = theano.shared(parameters['W_ff'], name='W_ff')
            b = theano.shared(parameters['b'], name='b')

        initial = U.create_shared(U.initial_weights(output_size))
        self.is_backward = is_backward

        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        def step(in_t, out_tminus1):
            return self.activation_fn(T.dot(out_tminus1, W_ff) + T.dot(in_t, W_if) + b)

        self.output, _ = theano.scan(
            step,
            sequences=[inputs],
            outputs_info=[initial],
            go_backwards=self.is_backward
        )

        self.params = [W_if, W_ff, b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params


class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):

        if parameters is None:
            W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            W = theano.shared(parameters['W'], name='W')
            b = theano.shared(parameters['b'], name='b')

        self.output = T.nnet.softmax(T.dot(inputs, W) + b)
        self.params = [W, b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params



# Courtesy of https://github.com/rakeshvar/rnn_ctc
# With T.eye() removed for k!=0 (not implemented on GPU for k!=0)
class CTCLayer():
    def __init__(self, inpt, labels, blank, batch_size):
        '''
        Recurrent Relation:
        A matrix that specifies allowed transistions in paths.
        At any time, one could
        0) Stay at the same label (diagonal is identity)
        1) Move to the next label (first upper diagonal is identity)
        2) Skip to the next to next label if
            a) next label is blank and
            b) the next to next label is different from the current
            (Second upper diagonal is product of conditons a & b)
        '''
        n_labels = labels[0].shape[0]
        big_I = T.cast(T.eye(n_labels+2), 'float64')
        recurrence_relation = T.cast(T.eye(n_labels), 'float64') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float64')
        recurrence_relation = T.cast(recurrence_relation, 'float64')


        inpt = T.reshape(inpt, (batch_size, inpt.shape[0]/batch_size, 30))

        def step(input, label):
            '''
            Forward path probabilities
            '''
            pred_y = input[:, label]

            probabilities, _ = theano.scan(
                lambda curr, prev: curr * T.dot(prev, recurrence_relation),
                sequences=[pred_y],
                outputs_info=[T.cast(T.eye(n_labels)[0], 'float64')]
            )
            return -T.log(T.sum(probabilities[-1, -2:]))

        probs, _ = theano.scan(
            step,
            sequences=[inpt, labels]
        )

        self.cost = T.mean(probs)
        self.params = []


class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality, data_x, data_y, max_len, params=None, batch_size=100, learning_rate=0.01, momentum=.25):
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        self.max_len = max_len

        input_stack = T.fmatrix('input_seq')
        label_stack = T.imatrix('label')
        index = T.iscalar()  # index to the sample

        if params is None:
            ff1 = FeedForwardLayer(input_stack, self.input_dimensionality, 2000)
            ff2 = FeedForwardLayer(ff1.output, 2000, 1000)
            ff3 = FeedForwardLayer(ff2.output, 1000, 500)
            rf = RecurrentLayer(ff3.output, 500, 250, False)     # Forward layer
            rb = RecurrentLayer(ff3.output, 500, 250, True)      # Backward layer
            s = SoftmaxLayer(T.concatenate((rf.output, rb.output), axis=1), 2*250, self.output_dimensionality)

        else:
            ff1 = FeedForwardLayer(input_stack, self.input_dimensionality, 2000, params[0])
            ff2 = FeedForwardLayer(ff1.output, 2000, 1000, params[1])
            ff3 = FeedForwardLayer(ff2.output, 1000, 500, params[2])
            rf = RecurrentLayer(ff3.output, 500, 250, False, params[3])     # Forward layer
            rb = RecurrentLayer(ff3.output, 500, 250, True, params[4])      # Backward layer
            s = SoftmaxLayer(T.concatenate((rf.output, rb.output), axis=1), 2*250, self.output_dimensionality, params[5])


        ctc = CTCLayer(s.output, label_stack, self.output_dimensionality-1, batch_size)
        
        updates = []
        for layer in (s, rb, rf, ff3, ff2, ff1):
            for p in layer.params:
                #param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                #grad = T.grad(ctc.cost, p)
                #updates.append((p, p - learning_rate * param_update))
                #updates.append((param_update, momentum * param_update + (1. - momentum) * grad))
                updates.append((p, p - learning_rate*T.grad(ctc.cost, p)))

        self.trainer = theano.function(
            inputs=[index],
            outputs=[ctc.cost],
            updates=updates,
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((self.max_len*batch_size, 240)),
                label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )
        self.tester = theano.function(
            inputs=[index],
            outputs=[s.output],
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((self.max_len*batch_size, 240)),
            }
        )

    def dump(self, f_path):
        f = file(f_path, 'wb')
        for obj in [self.ff1.get_parameters(), self.ff2.get_parameters(), self.ff3.get_parameters(), self.rf.get_parameters(), self.rb.get_parameters(), self.s.get_parameters()]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

class Network:
    def __init__(self):
        self.nn = None

    def create_network(self, input_dimensionality, output_dimensionality, data_x, data_y, max_len, batch_size=50, learning_rate=0.01, momentum=.25):
        self.nn = BRNN(input_dimensionality, output_dimensionality, data_x, data_y, max_len, params=None, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def load_network(self, path, input_dimensionality, output_dimensionality, data_x, data_y, max_len, batch_size=50, learning_rate=0.01, momentum=.25):
        f = file(path, 'rb')
        parameters = []
        for i in np.arange(6):
            parameters.append(pickle.load(f))
        f.close()

        for p in parameters:
            print type(p)

        self.nn = BRNN(input_dimensionality, output_dimensionality, data_x, data_y, max_len, params=parameters, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def dump_network(self, path):
        if self.nn is None:
            return False

        self.nn.dump(path)