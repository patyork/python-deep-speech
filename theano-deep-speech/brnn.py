__author__ = 'pat'
'''
Bidirectional Recurrent Neural Network
with Connectionist Temporal Classification (CTC)
  courtesy of https://github.com/shawntan/rnn-experiment
  courtesy of https://github.com/rakeshvar/rnn_ctc
implemented in Theano
'''

import theano
import theano.tensor as T
from theano_toolkit import utils as U
from theano_toolkit import updates
import numpy as np
import cPickle as pickle

#THEANO_FLAGS='device=cpu,floatX=float32'
theano.config.warn_float64='ignore'

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity='high'

def relu(x): return T.min(T.max(0, x), 20)

def clipped_relu(x): return np.min(np.max(0, x), 20)


class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)
        
        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')
        
        self.output = self.activation_fn(T.dot(inputs, self.W) + self.b)

        self.params = [self.W, self.b]
        
    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params



class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, is_backward=False, parameters=None):
    
        if parameters is None:
            self.W_if = U.create_shared(U.initial_weights(input_size, output_size), name='W_if')
            self.W_ff = U.create_shared(U.initial_weights(output_size, output_size), name='W_ff')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W_if = theano.shared(parameters['W_if'], name='W_if')
            self.W_ff = theano.shared(parameters['W_ff'], name='W_ff')
            self.b = theano.shared(parameters['b'], name='b')
            
        initial = U.create_shared(U.initial_weights(output_size))

        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)
        self.is_backward = is_backward

        def step(in_t, out_tminus1):
            return self.activation_fn(T.dot(out_tminus1, self.W_ff) + T.dot(in_t, self.W_if) + self.b)

        self.output, _ = theano.scan(
            step,
            sequences=[inputs],
            outputs_info=[initial],
            go_backwards=self.is_backward
        )

        self.params = [self.W_if, self.W_ff, self.b]
        
    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params
        

class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):
    
        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')

        self.output = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.params = [self.W, self.b]
        
    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

# Courtesy of https://github.com/rakeshvar/rnn_ctc
class CTCLayer():
    def __init__(self, inpt, labels, blank):
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
        n_labels = labels.shape[0]
        
        big_I = T.cast(T.eye(n_labels+2), 'float64')
        recurrence_relation = T.cast(T.eye(n_labels), 'float64') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float64')
        recurrence_relation = T.cast(recurrence_relation, 'float64')
        
        '''
        Forward path probabilities
        '''
        pred_y = inpt[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, prev: curr * T.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.cast(T.eye(n_labels)[0], 'float64')]
        )


        # Final Costs
        labels_probab = T.sum(probabilities[-1, -2:])
        self.cost = -T.log(labels_probab)
        self.params = []
        self.debug = probabilities.T


class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality, params=None, learning_rate=.01, momentum_rate=.25):
        inputs = T.matrix('input_seq')
        labels = T.ivector('labels')
        
        if params is None:
            self.ff1 = FeedForwardLayer(inputs, input_dimensionality, 2000)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2000, 1000)
            self.ff3 = FeedForwardLayer(self.ff2.output, 1000, 500)
            self.rf = RecurrentLayer(self.ff3.output, 500, 250, False)     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 500, 250, True)      # Backward layer
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output), axis=1), 2*250, output_dimensionality)
        else:
            self.ff1 = FeedForwardLayer(inputs, input_dimensionality, 2000, params[0])
            self.ff2 = FeedForwardLayer(self.ff1.output, 2000, 1000, params[1])
            self.ff3 = FeedForwardLayer(self.ff2.output, 1000, 500, params[2])
            self.rf = RecurrentLayer(self.ff3.output, 500, 250, False, params[3])     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 500, 250, True, params[4])      # Backward layer
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output), axis=1), 2*250, output_dimensionality, params[5])
            
        ctc = CTCLayer(self.s.output, labels, output_dimensionality-1)
        l2 = T.sum(self.ff1.W**2) + T.sum(self.ff2.W**2) + T.sum(self.ff3.W**2) + T.sum(self.s.W**2) + T.sum(self.rf.W_if**2) + T.sum(self.rf.W_ff**2) + T.sum(self.rb.W_if**2) + T.sum(self.rb.W_ff**2)

        updates = []
        for layer in (self.ff1, self.ff2, self.ff3, self.rf, self.rb, self.s):
            for p in layer.params:
                param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                grad = T.grad(ctc.cost - .005*l2, p)
                updates.append((p, p-learning_rate*param_update))
                updates.append((param_update, momentum_rate*param_update + (1. - momentum_rate)*grad))

        self.trainer = theano.function(
            inputs=[inputs, labels],
            outputs=[ctc.cost, self.s.output],
            updates=updates
        )

        self.validator = theano.function(
            inputs=[inputs, labels],
            outputs=[ctc.cost]
        )

        self.tester = theano.function(
            inputs=[inputs],
            outputs=[self.s.output]
        )
    
    def dump(self, f_path):
        f = file(f_path, 'wb')
        for obj in [self.ff1.get_parameters(), self.ff2.get_parameters(), self.ff3.get_parameters(), self.rf.get_parameters(), self.rb.get_parameters(), self.s.get_parameters()]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        
class Network:
    def __init__(self):
        self.nn = None

    def create_network(self, input_dimensionality, output_dimensionality, learning_rate=0.001, momentum=.99):
        self.nn = BRNN(input_dimensionality, output_dimensionality, params=None, learning_rate=learning_rate, momentum_rate=momentum)
        return self.nn

    def load_network(self, path, input_dimensionality, output_dimensionality, learning_rate=0.001, momentum=.99):
        f = file(path, 'rb')
        parameters = []
        for i in np.arange(6):
            parameters.append(pickle.load(f))
        f.close()

        for p in parameters:
            print type(p)

        self.nn = BRNN(input_dimensionality, output_dimensionality, params=parameters, learning_rate=learning_rate, momentum_rate=momentum)
        return self.nn

    def dump_network(self, path):
        if self.nn is None:
            return False

        self.nn.dump(path)
        
        

