import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "shakespeare.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        mapped_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(mapped_file)):

            # Read text file

            self.preprocess(input_file, vocab_file, mapped_file)

        else:

            # Load preprocessed file

            self.load_preprocessed(vocab_file, mapped_file)

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, mapped_file):

        with codecs.open(input_file, "r", encoding=self.encoding) as f:

            data = f.read()

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        with open(vocab_file, 'wb') as f:

            cPickle.dump(self.chars, f)

        self.mapped = np.array(list(map(self.vocab.get, data)))

        np.save(mapped_file, self.mapped)

    def load_preprocessed(self, vocab_file, mapped_file):

        with open(vocab_file, 'rb') as f:

            self.chars = cPickle.load(f)

        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.mapped = np.load(mapped_file)
        self.num_batches = int(self.mapped.size / (self.batch_size * self.seq_length))

    def create_batches(self):

        self.num_batches = int(self.mapped.size / (self.batch_size * self.seq_length))
        self.mapped = self.mapped[:self.num_batches * self.batch_size * self.seq_length]

        xdata = self.mapped
        ydata = np.copy(self.mapped)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):

        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]

        self.pointer += 1

        return x, y

    def reset_batch_pointer(self):

        self.pointer = 0
