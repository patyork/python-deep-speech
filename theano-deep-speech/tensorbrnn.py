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

THEANO_FLAGS='device=cpu,floatX=float32'
#theano.config.warn_float64='warn'

theano.config.optimizer = 'fast_run'
theano.config.exception_verbosity = 'high'
theano.config.on_unused_input = 'warn'

class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        W = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        self.output2 = self.activation_fn(T.dot(inputs, W) + b)

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

        self.output, _ = theano.scan(
            lambda in_t: theano.scan(
                lambda index, out_tminus1: self.activation_fn(T.dot(out_tminus1, W_ff) + T.dot(in_t[index], W_if) + b),
                sequences=[T.arange(inputs.shape[1])],
                outputs_info=[initial],
                go_backwards=is_backward
            ),
            sequences=[inputs]  # for each sample at time "t"
        )

        self.params = [W_if, W_ff, b]


class SoftmaxLayer:
    def __init__(self, forward_in, backward_in, input_size, output_size):
        Wf = U.create_shared(U.initial_weights(input_size, output_size))
        Wb = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        self.activations = T.dot(forward_in, Wf) + T.dot(backward_in, Wb) + b

        self.output, _ = theano.scan(
            lambda inpt: T.nnet.softmax(inpt),
            sequences=[self.activations]
        )

        self.params = [Wf, Wb, b]


class CTCLayer():
    def __init__(self, inpt, labels):
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
        n_labels = labels.shape[1]
        big_I = T.eye(n_labels+2)
        recurrence_relation = T.eye(n_labels) + big_I[2:, 1:-1] + big_I[2:, :-2] * (T.arange(n_labels) % 2)
        #recurrence_relation = T.cast(recurrence_relation, 'float64')

        # Stack the probabilities in the order of the modified sequence, l'
        pred_y1, _ = theano.scan(
            lambda ndex: inpt[ndex][:, labels[ndex]],
            sequences=[T.arange(inpt.shape[0])]
        )
        self.pred_y1 = pred_y1

        # Partition the probabilities into each time step
        pred_y, _ = theano.scan(
            lambda ndex: pred_y1[:, ndex, :],
            sequences=[T.arange(n_labels)]
        )
        self.pred_y = pred_y

        # Calculate the forward path probability
        probabilities, _ = theano.scan(
            lambda curr, prev: curr * T.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.cast(T.tile(T.eye(n_labels)[0], (200,)).reshape((200, n_labels)), 'float64')]
        )

        self.probabilities = probabilities
        self.cost = -T.log(T.mean(probabilities[-1, :, -2:])*probabilities.shape[1])

        self.params = []


class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality, data_x, data_y, batch_size=200, learning_rate=0.01, momentum=.25):
        input_stack = T.ftensor3('input_seq')
        #label = T.ivector('label')
        label_stack = T.imatrix('label')
        index = T.iscalar()  # index to the sample

        ff1 = FeedForwardLayer(input_stack, input_dimensionality, 200)
        ff2 = FeedForwardLayer(ff1.output, 200, 100)
        ff3 = FeedForwardLayer(ff2.output, 100, 50)
        rf = RecurrentLayer(ff3.output, 50, 25, False)     # Forward layer
        rb = RecurrentLayer(ff3.output, 50, 25, True)      # Backward layer
        s = SoftmaxLayer(rf.output, rb.output, 25, output_dimensionality)
        ctc = CTCLayer(s.output, label_stack)

        '''updates = []
        for layer in (ff1, ff2, ff3, rf, rb, s, ctc):
            for p in layer.params:
                param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                grad = T.grad(ctc.cost, p)
                updates.append((p, p - learning_rate * param_update))
                updates.append((param_update, momentum * param_update + (1. - momentum) * grad))
        '''
        '''
        self.tester = theano.function(
            inputs=[input_stack],
            outputs=[s.output]
        )

        self.debug = theano.function(
            inputs=[index],
            outputs=[ctc.mean_cost, ctc.probabilities],
            #updates=updates,
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((281*batch_size, 240)),
                label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )'''
        self.debugTest = theano.function(
            inputs=[index],
            outputs=[ctc.cost, s.output],
            #updates=updates,
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size],
                label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )