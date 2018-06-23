from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
 
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import *

import argparse
import logging
import numpy as np
import mmh3

class Dataset:
    
    def __init__(self, dataname, in_file, max_lines, out_file, cache, model):
        logging.debug('Using %s data from %s' % (dataname, in_file))
        self.dataname = dataname
        self.in_file = in_file
        self.max_lines = max_lines
        self.out_file = out_file
        self.cache = cache
        self.data = None
        self.yhats = None
        self.model = model
        
    def iterdata(self):
        it = data_iterator(self.in_file, self.max_lines, self.dataname, self.model)
        if self.cache:
            if self.data is None:
                logging.debug('Caching %s data' % (self.dataname))
                self.data = list(it)
                ys = [y for _, y in self.data]
                logging.debug('%s data: %d sentences, %d label types' % 
                             (self.dataname, len(ys), len(np.unique(ys))))
            return self.data
        else:
            return it
    

def data_iterator(in_file, max_lines, dataname, model):
    '''Read tab-separated sentences formatted as: 
    label <TAB> feat1 <SPACE> feat2 <SPACE> ... <SPACE> featn

    Args:
        in_file: Input file.
        max_lines: The maximum number of lines to read.
    
    Returns:
        A pair containing a list of instances and a list of labels.
    '''
    with open(in_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            splits = line.strip().split('\t')
            label, feat_strs = splits
            
            # Convert feature strings to hashed ints
            x = get_features(feat_strs, model)
            # Convert label to integer
            y = model.lookup_idx(label)
            
            yield x, y
            
            if max_lines is not None and i == max_lines - 1:
                break

def get_features(feat_strs, model):
    f = []
    for feat_str in feat_strs.split(' '):
        if feat_str == '': continue
        f.append(get_feature(feat_str, model))
    # Include a constant (aka. bias) feature in each instance
    f.append(get_feature('bias', model))
    return np.array(f, dtype=np.int32)

def get_feature(feat_str, model):
    # The feature string may be unicode, but MurmurHash3 expects ASCII encoded strings.
    return mmh3.hash(feat_str.encode('ascii', 'xmlcharrefreplace')) % model.num_features

def write_labels(out_file, yhats, model, dataname):
    logging.info('Saving %s predictions to %s' % (dataname, out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        for i in range(len(yhats)):
            f.write(model.lookup_label(yhats[i]))
            f.write('\n')


class Model:
    
    def __init__(self, num_features, labels_path):
        self.num_features = num_features
        self.label2idx = {}
        self.idx2label = []
        self.load_labels(labels_path)
        self.num_labels = len(self.label2idx)
        self.params = None
        
    def lookup_idx(self, label):
        return self.label2idx[label]

    def lookup_label(self, idx):
        return self.idx2label[idx]
    
    def load_labels(self, labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line.strip()
                if label in self.label2idx:
                    raise Exception("Label (%s) appears multiple times in file (%s)." % (label, labels_path))
                idx = len(self.label2idx)
                self.label2idx[label] = idx
                self.idx2label.append(label)
    

def learn(train, dev, model, num_epochs=1, dev_iters=None):
    '''Learn a model on the given training data, using the development for validation.

    Args:
        train: A Dataset object for the training data.
        dev: A Dataset object for the dev data.
        model: A Model object.
    '''
    logging.debug('Training...')
    
    model.params = np.zeros(shape=(model.num_labels, model.num_features), dtype=np.float)
    t = 0
    next_print = 1
    for epoch in range(num_epochs):        
        # Run SGD for one pass through the train data.
        for x, y in train.iterdata():
            sgd_step(model.params, x, y, t)
            t += 1
            if t == next_print:
                next_print *= 2
                logging.info('Epoch: %d Iteration: %d Features: %d' % 
                             (epoch, t, len(x)))

            if dev_iters is not None and t % dev_iters == 0:
                # Validate on the dev data.
                _, accuracy = predict_and_eval(model.params, dev)
                logging.info('Epoch: %d Iteration: %d Features: %d Accuracy on dev: %.2f' % 
                             (epoch, t, len(x), accuracy))
        # Default to validating at the end of each epoch.
        if dev_iters is None and epoch != num_epochs - 1:
            # Validate on the dev data.
            _, accuracy = predict_and_eval(model.params, dev)
            logging.info('Epoch: %d Iteration: %d Features: %d Accuracy on dev: %.2f' % 
                         (epoch, t, len(x), accuracy))

def sgd_step(params, x, y, t):
    learning_rate = 0.1
    num_labels = params.shape[0]
    p = get_probabilities(params, x)
    for yprime in range(num_labels):
        for feat in x:
            if yprime == y: v = 1
            else:           v = 0
            params[yprime, feat] += learning_rate * (v - p[yprime])

def get_probabilities(params, x):
    p = sparse_mult(params, x)
    return softmax(p, inplace=True)

def sparse_mult(A, x):
    '''Multiples a dense matrix A times a sparse integer vector x.
    
    Args:
        A: Dense 2D numpy array.
        x: Sparse vector represented as a list of indices. For each index found in the list,
            its value is assumed to be the number of occurrences of that index. Indices that
            are absent have value 0.
    '''
    return A[:, x].sum(axis=1)

def softmax(v, inplace=False):
    '''Applies the softmax function to the dense vector v.
    
    Args:
        v: The numpy array representing the dense vector.
        inplace: Whether the computation should overwrite values in v, or return
            a new array.
    '''
    if inplace: p = v
    else: p = v.copy()
    # Exponentiate each element.
    np.exp(p, p)
    # Normalize the distribution.
    p /= np.sum(p)
    return p

def predict_and_eval(params, dataset):
    '''Predicts a label for each of the instances in the dataset and computes 
    the accuracy of the predicted labels.

    Args:
        params: The trained model parameters.
        dataset: A Dataset object containing the instances and true (gold) labels.

    Returns:
        A pair containing the predicted labels and their accuracy.
    '''
    num_correct = 0
    yhats = []
    for x, y in dataset.iterdata():
        # Predict
        p = get_probabilities(params, x)
        yhat = np.argmax(p)
        yhats.append(yhat)
        # Evaluate
        if yhat == y: 
            num_correct += 1
    accuracy = float(num_correct) / len(yhats) 
    return yhats, accuracy

def summarize(model):
    logging.info('Number of total labels: %d' % (model.num_labels))
    logging.info('Label set: %s' % (sorted(model.label2idx.keys())))
    logging.info('Max features: %d' % (model.num_features))
    
def main(args):
    '''Trains a model. '''
    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(levelname)s - %(message)s')

    model = Model(args.num_features, args.labels)

    # Cache the training data if we will be doing multiple epochs
    cache_train = (args.num_epochs > 1)
    # Cache dev always.
    cache_dev = True
    
    train = Dataset('train', args.train, args.train_max, args.train_out, cache_train, model)
    dev = Dataset('dev', args.dev, args.dev_max, args.dev_out, cache_dev, model)
    test = Dataset('test', args.test, args.test_max, args.test_out, False, model)

    summarize(model)
    
    learn(train, dev, model,
          args.num_epochs, args.dev_iters)

    datasets = [train, dev, test]
    for d in datasets:
        yhats, accuracy = predict_and_eval(model.params, d)
        logging.info('Accuracy on %s: %.2f' % (d.dataname, accuracy))
        if d.out_file is not None:
            write_labels(d.out_file, yhats, model, d.dataname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data options
    parser.add_argument('--train', required=True, help='Train file')
    parser.add_argument('--dev',   required=True, help='Dev file')
    parser.add_argument('--test',  required=True, help='Test file')
    parser.add_argument('--train_max', type=int, help='Max instances to read from Train')
    parser.add_argument('--dev_max',   type=int, help='Max instance to read from Dev')
    parser.add_argument('--test_max',  type=int, help='Max instances to read from Test')
    parser.add_argument('--train_out', help='Train output file')
    parser.add_argument('--dev_out',   help='Dev output file')
    parser.add_argument('--test_out',  help='Test output file')
    
    # Model options
    parser.add_argument('--num_features', type=int, default=1000000, help='Maximum number of features')
    parser.add_argument('--labels', required=True, help='Labels file')
    
    # Training options
    parser.add_argument('--num_epochs', type=int, default=1, help='Num training passes')
    parser.add_argument('--dev_iters', type=int, help='Num iterations between validation on dev')
    
    args = parser.parse_args()

    main(args)
