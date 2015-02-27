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

#THEANO_FLAGS='device=cpu,floatX=float32'
#theano.config.warn_float64='warn'

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity='high'


class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        W = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        self.output, _ = theano.scan(
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
            return self.activation_fn(T.dot(out_tminus1, W_ff) + T.dot(in_t, W_if) + b)

        self.output, _ = theano.scan(
            step,
            sequences=[inputs],
            outputs_info=[initial],
            go_backwards=is_backward
        )

        self.params = [W_if, W_ff, b]

class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size):
        W = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        self.output = T.nnet.softmax(T.dot(inputs, W) + b)
        self.params = [W, b]


# Courtesy of https://github.com/rakeshvar/rnn_ctc
# With T.eye() removed for k!=0 (not implemented on GPU for k!=0)
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
        n_labels = 79#labels[0].shape[0]
        big_I = T.cast(T.eye(n_labels+2), 'float64')
        recurrence_relation = T.cast(T.eye(n_labels), 'float64') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float64')
        recurrence_relation = T.cast(recurrence_relation, 'float64')


        inpt = T.reshape(inpt, (1000, 291, 30))

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
    def __init__(self, input_dimensionality, output_dimensionality, data_x, data_y, batch_size=1000, learning_rate=0.01, momentum=.25):
        input_stack = T.fmatrix('input_seq')
        # label = T.ivector('label')
        label_stack = T.imatrix('label')
        index = T.iscalar()  # index to the sample

        ff1 = FeedForwardLayer(input_stack, input_dimensionality, 200)
        ff2 = FeedForwardLayer(ff1.output, 200, 100)
        ff3 = FeedForwardLayer(ff2.output, 100, 50)
        rf = RecurrentLayer(ff3.output, 50, 25, False)     # Forward layer
        rb = RecurrentLayer(ff3.output, 50, 25, True)      # Backward layer
        s = SoftmaxLayer(T.concatenate((rf.output, rb.output), axis=1), 2*25, output_dimensionality)
        ctc = CTCLayer(s.output, label_stack, output_dimensionality-1)
        
        '''updates = []
        for layer in (ff1, ff2, ff3, rf, rb, s):
            for p in layer.params:
                param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                grad = T.grad(ctc.cost, p)
                updates.append((p, p - learning_rate * param_update))
                updates.append((param_update, momentum * param_update + (1. - momentum) * grad))

        ''''''
        self.trainer = theano.function(
            inputs=[index],
            #outputs=[s.output],
            outputs=[ctc.cost, s.output],
            updates=updates,
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((209*batch_size, 240)),
                label_stack: data_y[index*batch_size:(index+1)*batch_size]
                #label: data_y[index]
            }
        )


        self.validator = theano.function(
            inputs=[input_stack, label_stack],
            #outputs=[ctc.cost]
        )


        self.tester = theano.function(
            inputs=[input_stack],
            outputs=[s.output]
        )
        '''
        self.debug = theano.function(
            inputs=[index],
            outputs=[ctc.cost],
            #updates=updates,
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((291*batch_size, 240)),
                label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )
        self.debugTest = theano.function(
            inputs=[index],
            outputs=[s.output],
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((291*batch_size, 240)),
                #label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )
