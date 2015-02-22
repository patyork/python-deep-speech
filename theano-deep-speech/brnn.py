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

#THEANO_FLAGS='device=cpu,floatX=float32'
theano.config.warn_float64='warn'

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity='high'

def relu(x): return T.min(T.max(0, x), 20)

def clipped_relu(x): return np.min(np.max(0, x), 20)


class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        W = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        #def step(in_t, out_tminus1):
        #    return T.tanh(T.dot(int_t, W) + b)

        self.output, _ = theano.scan(
            #lambda element: T.nnet.softplus(T.dot(element, W) + b),
            lambda element: self.activation_fn(T.dot(element, W) + b),
            sequences=[inputs]
        )

        self.params = [W, b]



class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, is_backward=False):
        W_if = U.create_shared(U.initial_weights(input_size, output_size))
        W_ff = U.create_shared(U.initial_weights(output_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))
        initial = U.create_shared(U.initial_weights(output_size))

        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        def step(in_t, out_tminus1):
            #return T.nnet.softplus(T.dot(out_tminus1, W_ff) + T.dot(in_t, W_if) + b)
            return self.activation_fn(T.dot(out_tminus1, W_ff) + T.dot(in_t, W_if) + b)

        self.output, _ = theano.scan(
            step,
            sequences=[inputs],
            outputs_info=[initial],
            go_backwards=is_backward
        )

        self.params = [W_if, W_ff, b, initial]

class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size):
        W = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        self.output = T.nnet.softmax(T.dot(inputs, W) + b)
        self.params = [W, b]


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
        #labels2 = T.concatenate((labels, [blank, blank]))
        #sec_diag = T.neq(labels2[:-2], labels2[2:]) * T.eq(labels2[1:-1], blank)
        n_labels = labels.shape[0]

        #recurrence_relation = \
        #       T.eye(n_labels) + \
        #       T.eye(n_labels, k=1) + \
        #       T.eye(n_labels, k=2) * sec_diag.dimshuffle((0, 'x'))
        
        big_I = T.cast(T.eye(n_labels+2), 'float32')
        recurrence_relation = T.cast(T.eye(n_labels), 'float32') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float32')
        recurrence_relation = T.cast(recurrence_relation, 'float32')
        '''
        Forward path probabilities
        '''
        pred_y = inpt[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, prev: curr * T.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.cast(T.eye(n_labels)[0], 'float32')]
        )


        '''
        pred_y = T.log(pred_y)                          # Probabilities in log scale
        log_scale_probabilities, _ = theano.scan(
            lambda curr, prev: curr * T.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.eye(n_labels)[0]]    # initial state in log scale
        )'''

        # Final Costs
        labels_probab = T.sum(probabilities[-1, -2:])
        self.cost = -T.log(labels_probab)
        self.params = []
        self.debug = probabilities.T

class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality):
        inputs = T.matrix('input_seq')
        labels = T.ivector('labels')
        momentum = .25

        ff1 = FeedForwardLayer(inputs, input_dimensionality, 200)
        ff2 = FeedForwardLayer(ff1.output, 200, 100)
        ff3 = FeedForwardLayer(ff2.output, 100, 50)
        rf = RecurrentLayer(ff3.output, 50, 25, False)     # Forward layer
        rb = RecurrentLayer(ff3.output, 50, 25, True)      # Backward layer
        s = SoftmaxLayer(T.concatenate((rf.output, rb.output), axis=1), 2*25, output_dimensionality)
        ctc = CTCLayer(s.output, labels, output_dimensionality-1)

        updates = []
        for layer in (ff1, ff2, ff3, rf, rb, s, ctc):
            for p in layer.params:
                param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                grad = T.grad(ctc.cost, p)
                updates.append((p, p-.01*param_update))
                updates.append((param_update, momentum*param_update + (1. - momentum)*grad))

        self.trainer = theano.function(
            inputs=[inputs, labels],
            outputs=[ctc.cost, s.output],
            updates=updates
        )

        self.validator = theano.function(
            inputs=[inputs, labels],
            outputs=[ctc.cost]
        )

        self.tester = theano.function(
            inputs=[inputs],
            outputs=[s.output]
        )
