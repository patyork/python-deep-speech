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
    def __init__(self, inputs, input_size, output_size):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        W = U.create_shared(U.initial_weights(input_size, output_size))
        b = U.create_shared(U.initial_weights(output_size))

        self.output, _ = theano.scan(
            lambda element: self.activation_fn(T.dot(element, W) + b),
            sequences=[inputs]
        )

        self.params = [W, b]

    def __getstate__(self):
        return (self.W.get_value(), self.b.get_value())

    def __setstate__(self, state):
        #W, b = state
        self.W.set_value(state[0])
        self.b.set_value(state[1])



class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, is_backward=False):
        self.W_if = U.create_shared(U.initial_weights(input_size, output_size))
        self.W_ff = U.create_shared(U.initial_weights(output_size, output_size))
        self.b = U.create_shared(U.initial_weights(output_size))
        self.initial = U.create_shared(U.initial_weights(output_size))
        self.is_backward = is_backward

        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        def step(in_t, out_tminus1):
            return self.activation_fn(T.dot(out_tminus1, self.W_ff) + T.dot(in_t, self.W_if) + self.b)

        self.output, _ = theano.scan(
            step,
            sequences=[inputs],
            outputs_info=[self.initial],
            go_backwards=self.is_backward
        )

        self.params = [self.W_if, self.W_ff, self.b]

    def __getstate__(self):
        return (self.W_if.get_value(), self.W_ff.get_value(), self.b.get_value(), self.initial.get_value(), self.is_backward)

    def __setstate__(self, state):
        W_if, W_ff, b, initial, is_back = state
        self.W_if.set_value(W_if)
        self.W_ff.set_value(W_ff)
        self.b.set_value(b)
        self.initial.set_value(initial)
        self.is_backward = is_back

class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size):
        self.W = U.create_shared(U.initial_weights(input_size, output_size))
        self.b = U.create_shared(U.initial_weights(output_size))

        self.output = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.params = [self.W, self.b]

    def __getstate__(self):
        return (self.W.get_value(), self.b.get_value())

    def __setstate__(self, state):
        W, b = state
        self.W.set_value(W)
        self.b.set_value(b)


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
    def __init__(self, input_dimensionality, output_dimensionality, data_x, data_y, batch_size=100, learning_rate=0.01, momentum=.25):
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        input_stack = T.fmatrix('input_seq')
        # label = T.ivector('label')
        label_stack = T.imatrix('label')
        index = T.iscalar()  # index to the sample

        self.ff1 = FeedForwardLayer(input_stack, self.input_dimensionality, 2000)
        self.ff2 = FeedForwardLayer(self.ff1.output, 2000, 1000)
        self.ff3 = FeedForwardLayer(self.ff2.output, 1000, 500)
        self.rf = RecurrentLayer(self.ff3.output, 500, 250, False)     # Forward layer
        self.rb = RecurrentLayer(self.ff3.output, 500, 250, True)      # Backward layer
        self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output), axis=1), 2*250, self.output_dimensionality)
        ctc = CTCLayer(self.s.output, label_stack, self.output_dimensionality-1, batch_size)
        
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
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((256*batch_size, 240)),
                label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )
        self.debugTest = theano.function(
            inputs=[index],
            outputs=[self.s.output],
            givens={
                input_stack: data_x[index*batch_size:(index+1)*batch_size].reshape((256*batch_size, 240)),
                #label_stack: data_y[index*batch_size:(index+1)*batch_size]
            }
        )

    def dump(self, f_path):
        f = file(f_path, 'wb')
        for obj in [self.ff1, self.ff2, self.ff3, self.rf, self.rb, self.s]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, f_path):
        f = file(f_path, 'rb')
        loaded = []
        for i in np.arange(6):
            loaded.append(pickle.load(f))
        f.close()

        self.ff1v = loaded[0]
        self.ff2 = loaded[1]
        self.ff3 = loaded[2]
        self.rf = loaded[3]
        self.rb = loaded[4]
        self.s = loaded[5]
