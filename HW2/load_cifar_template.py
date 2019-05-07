import pickle
import numpy as np
from tqdm import trange

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):

    """
    Args:
        folder_path: the directory contains data files
        batch_id: training batch id (1,2,3,4,5)
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    ###load batch using pickle###
    with open(folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
    # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    ###fetch features using the key ['data']###
    features = batch['data']
    ###fetch labels using the key ['labels']###
    labels = batch['labels']

    return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):

    """
    Args:
        folder_path: the directory contains data files
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    ###load batch using pickle###
    with open(folder_path + '/test_batch', mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
    ###fetch features using the key ['data']###
    features = batch['data']
    ###fetch labels using the key ['labels']###
    labels = batch['labels']

    return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():

    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
    Args:
        features: a numpy array with shape (10000, 3072)
    Return:
        features: a numpy array with shape (10000,32,32,3)
    """
    features_reshaped = []

    for row in features:

        arr = np.array([])

        r = row[0:1024].reshape(1024,1)
        g = row[1024:2048].reshape(1024,1)
        b = row[2048:3072].reshape(1024,1)

        arr = np.concatenate((r,g,b),axis=1)

        features_reshaped.append(arr.reshape(32,32,3))

    return np.stack(features_reshaped)

#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
    """
    Args:
        folder_path: directory that contains data files
        batch_id: the specific number of batch you want to explore.
        data_id: the specific number of data example you want to visualize
    Return:
        None

    Descrption: 
        1)You can print out the number of images for every class. 
        2)Visualize the image
        3)Print out the minimum and maximum values of pixel 
    """
    pass

#Step 6: define a function that does min-max normalization on input
def normalize(x):
    """
    Args:
        x: features, a numpy array
    Return:
        x: normalized features
    """
    normalized = []

    for image in x:

        min_val = np.min(image)
        max_val = np.max(image)

        image = (image-min_val) / (max_val-min_val)
        normalized.append(image)
    
    return np.stack(normalized)

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    """
    Args:
        x: a list of labels
    Return:
        a numpy array that has shape (len(x), # of classes)
    """
    encoded = np.zeros((len(x), 10))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
    """
    Args:
        features: numpy array
        labels: a list of labels
        filename: the file you want to save the preprocessed data
    """
    features = features_reshape(features)
    features = normalize(features)
    labels = one_hot_encoding(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    """
    Args:
        folder_path: the directory contains your data files
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):

        print('loading training batch')

        features, labels = load_training_batch(folder_path, batch_i)
        
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch

        print('preprocessing and saving')
        preprocess_and_save(features[:-index_of_validation], labels[:-index_of_validation], 
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    print('preprocessing validation dataset')
    preprocess_and_save(np.stack(valid_features), np.stack(valid_labels),
                         'preprocess_validation.p')


    print('preprocessing and saving testing dataset')
    # load the test dataset
    test_features, test_labels = load_testing_batch(folder_path)

    # Preprocess and Save all testing data
    preprocess_and_save(np.array(test_features), np.array(test_labels),
                         'preprocess_testing.p')

#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
    """
    Args:
        features: features for one batch
        labels: labels for one batch
        mini_batch_size: the mini-batch size you want to use.
    Hint: Use "yield" to generate mini-batch features and labels
    """
    for start_idx in trange(0, len(features) - mini_batch_size + 1, mini_batch_size):

        excerpt = slice(start_idx, start_idx + mini_batch_size)
        
        yield features[excerpt], labels[excerpt]

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
    """
    Args:
        batch_id: the specific training batch you want to load
        mini_batch_size: the number of examples you want to process for one update
    Return:
        mini_batch(features,labels, mini_batch_size)
    """
    file_name = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(file_name, 'rb')

    return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():

    file_name = 'preprocess_validation.p'
    features,labels = pickle.load(file_name, 'rb')

    return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = 'preprocess_testing.p'
    features,label = pickle.load(file_name, 'rb')
    
    return mini_batch(features,labels,test_mini_batch_size)

