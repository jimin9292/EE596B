import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""
class LSTM():

    def __init__(self, sample = False):

        rnn_size = 128 
        batch_size = 60 
        seq_length = 50 
        num_layers = 2 
        vocab_size = 68
        grad_clip = 5. # to prevent exploding gradient

        if sample:
            
            batch_size = 1
            seq_length = 1

        # RNN architecture definition #######################################################################################################################################

        rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)

        self.stacked_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * num_layers)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="input_data")
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")

        self.initial_state = self.stacked_cell.zero_state(batch_size, tf.float32) 

        with tf.variable_scope('scope1'):

            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size]) 

            # Vector of 128 dim for each character 
            embedding = tf.get_variable("embedding", [vocab_size, rnn_size]) 
            inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # compute outputs/last state
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.stacked_cell, loop_function=None, scope='scope1')
        
        # Reshape it back to proper dim (60x50x128)
        output = tf.reshape(tf.concat(outputs,1), [-1, rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # Loss/Optimizer ###################################################################################################################################################
        # Weighted cross-entropy loss for a sequence of logits (per example)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.targets, [-1])], [tf.ones([batch_size * seq_length])], vocab_size)
        # Use reduce sum
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),grad_clip)
        # learning rate is adaptive/coming from outside
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):

        state = sess.run(self.stacked_cell.zero_state(1, tf.float32))

        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]

        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)

            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)

            else: 
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred

        return ret