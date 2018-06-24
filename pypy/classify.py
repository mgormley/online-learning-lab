#!/usr/bin/env pypy3
import argparse
import logging
import array
import math

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
                             (self.dataname, len(ys), len(set(ys))))
            return self.data
        else:
            return it
    

def data_iterator(in_file, max_lines, dataname, model):
    """Read tab-separated sentences formatted as: 
    label <TAB> feat1 <SPACE> feat2 <SPACE> ... <SPACE> featn

    Args:
        in_file: Input file.
        max_lines: The maximum number of lines to read.
    
    Returns:
        A pair containing a list of instances and a list of labels.
    """
    with open(in_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            splits = line.strip().split('\t')
            label, feat_strs = splits
            
            # Convert feature strings to hashed ints
            x = get_features(feat_strs, model.num_features)
            # Convert label to integer
            y = model.lookup_idx(label)
            
            yield x, y
            
            if max_lines is not None and i == max_lines - 1:
                break

def get_features(feat_strs, num_features):
    f = []
    for feat_str in feat_strs.split(' '):
        if feat_str == '': continue
        f.append(get_feature(feat_str, num_features))
    # Include a constant (aka. bias) feature in each instance
    f.append(get_feature('bias', num_features))
    return f

def get_feature(feat_str, num_features):
    # The feature string may be unicode, but MurmurHash3 expects ASCII encoded strings.
    #return mmh3.hash(feat_str.encode('ascii', 'xmlcharrefreplace')) % num_features
    return hash(feat_str) % num_features

def write_labels(out_file, yhats, model, dataname):
    logging.info('Saving %s predictions to %s' % (dataname, out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        for i in range(len(yhats)):
            f.write(model.lookup_label(yhats[i]))
            f.write('\n')

def write_stats(out_file, stats):
    logging.info('Writing stats to %s' % (out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        for k,v in stats.items():
            f.write(str(k))
            f.write('\t')
            f.write(str(v))
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

def learn(train, dev, model, num_epochs=1, dev_iters=None, learning_rate=1.0):
    """Learn a model on the given training data, using the development for validation.

    Args:
        train: A Dataset object for the training data.
        dev: A Dataset object for the dev data.
        model: A Model object.
    """
    logging.debug('Training...')
    
    model.params = zeros_float(model.num_labels * model.num_features)
    t = 0
    next_print = 1
    for epoch in range(num_epochs):        
        # Run SGD for one pass through the train data.
        for x, y in train.iterdata():
            sgd_step(model.params, model.num_labels, model.num_features, x, y, t, learning_rate)
            t += 1
            if t == next_print:
                next_print *= 2
                logging.info('Epoch: %d Iteration: %d Features: %d' % 
                             (epoch, t, len(x)))

            if dev_iters is not None and t % dev_iters == 0:
                # Validate on the dev data.
                _, accuracy = predict_and_eval(model, dev)
                logging.info('Epoch: %d Iteration: %d Features: %d Accuracy on dev: %.2f' % 
                             (epoch, t, len(x), accuracy))
        # Default to validating at the end of each epoch.
        if dev_iters is None and epoch != num_epochs - 1:
            # Validate on the dev data.
            _, accuracy = predict_and_eval(model, dev)
            logging.info('Epoch: %d Iteration: %d Features: %d Accuracy on dev: %.2f' % 
                         (epoch, t, len(x), accuracy))

def sgd_step(params, num_labels, num_features, x, y, t, learning_rate):
    """Take one stochastic gradient descent step."""
    p = get_probabilities(params, num_labels, num_features, x)
    for yprime in range(num_labels):
        for feat in x:
            if yprime == y: v = 1
            else:           v = 0
            params[yprime * num_features + feat] += learning_rate * (v - p[yprime])

def get_probabilities(params, num_labels, num_features, x):
    """Compute the probabilities p(y | x) for all values y."""
    # Get scores
    p = zeros_float(num_labels)
    for y in range(len(p)):
        p[y] = fastdot(params, x, y, num_features)
    # Exponentiate then normalize
    softmax(p)
    return p
    
def fastdot(params, x, y, num_features):
    """Compute the score w^T f(x,y), where w is the parameter vector and 
    f(x,y) is a feature vector.
    """
    dot = 0
    for feat in x:
        dot += params[y * num_features + feat] 
    return dot

def softmax(v):
    """Applies the softmax function to the dense vector v. (in place)
    Uses the exp-normalize trick.
    Args:
        v: The numpy array representing the dense vector.
    """
    max_score = max(v);
    Z = 0
    # Compute shifted scores and exponentiate.
    for y in range(len(v)):
        v[y] = math.exp(v[y] - max_score)
        Z += v[y]
    # Normalize the shifted scores.
    for y in range(len(v)):
        v[y] /= Z
    return v

def predict_and_eval(model, dataset):
    """Predicts a label for each of the instances in the dataset and computes 
    the accuracy of the predicted labels.

    Args:
        model: The model
        dataset: A Dataset object containing the instances and true (gold) labels.

    Returns:
        A pair containing the predicted labels and their accuracy.
    """
    num_correct = 0
    yhats = []
    for x, y in dataset.iterdata():
        # Predict
        p = get_probabilities(model.params, model.num_labels, model.num_features, x)
        yhat = argmax(p)
        yhats.append(yhat)
        # Evaluate
        if yhat == y: 
            num_correct += 1
    accuracy = float(num_correct) / len(yhats) * 100.0 
    return yhats, accuracy

def argmax(v):
    """Get the index i with value that maximizes v[i]."""
    maxval = v[0]
    argmax = 0
    for i in range(1, len(v)):
        if v[i] > maxval:
            maxval = v[i]
            argmax = i
    return argmax

def zeros_float(length):
    """Create a list of n floats, each with value 0.0."""
    return [0.0] * length

def summarize(model):
    logging.info('Number of total labels: %d' % (model.num_labels))
    logging.info('Label set: %s' % (sorted(model.label2idx.keys())))
    logging.info('Max features: %d' % (model.num_features))
    
def main(args):
    """Trains a model."""
    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(levelname)s - %(message)s')
    
    stats = {}
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
          args.num_epochs, args.dev_iters, args.learning_rate)

    datasets = [train, dev, test]
    for d in datasets:
        yhats, accuracy = predict_and_eval(model, d)
        logging.info('Accuracy on %s: %.2f' % (d.dataname, accuracy))
        if d.out_file is not None:
            write_labels(d.out_file, yhats, model, d.dataname)
        stats[d.dataname+'_accuracy'] = accuracy
    
    if args.stats is not None:
        write_stats(args.stats, stats)

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
    parser.add_argument('--stats',  help='Stats output file')
    
    # Model options
    parser.add_argument('--num_features', type=int, default=1000000, help='Maximum number of features')
    parser.add_argument('--labels', required=True, help='Labels file')
    
    # Training options
    parser.add_argument('--num_epochs', type=int, default=1, help='Num training passes')
    parser.add_argument('--dev_iters', type=int, help='Num iterations between validation on dev')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    
    args = parser.parse_args()

    main(args)
