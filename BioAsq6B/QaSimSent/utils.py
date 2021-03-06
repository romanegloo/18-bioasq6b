"""utilities (and data structure) for qa_proximity model"""
import logging
import json
import unicodedata
import torch
from torch.nn.modules.module import _addindent
from torch.utils.data import Dataset
import numpy as np
import pickle
import time

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


# ------------------------------------------------------------------------------
# PyTorch Dataset class for qa_proximity model
# ------------------------------------------------------------------------------

class QaProxDataset(Dataset):
    def __init__(self, conf, examples, word_dict, feature_dict=None, idf=None):
        self.conf = conf
        self.ex = examples
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        if idf is not None:
            # Read idf file
            self.idf = pickle.load(open(idf, 'rb'))
        else:
            self.idf = None

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        ex = self.ex[idx]
        # Index words
        context = torch.LongTensor([self.word_dict[w] for w in ex['context']])
        question = torch.LongTensor([self.word_dict[w] for w in ex['question']])
        # Create feature vector
        question_types = ['yesno', 'factoid', 'list', 'summary']
        feat_q = torch.zeros(len(question_types))
        feat_q[question_types.index(ex['type'])] = 1

        feature_len = len(self.feature_dict) if self.feature_dict else 0
        feat_c = None
        if feature_len > 0:
            feat_c = torch.zeros(len(ex['context']), feature_len)
            # Feature POS or NER
            for key in ['pos', 'ner']:
                for i, w in enumerate(ex[key]):
                    if key + '=' + w in self.feature_dict:
                        feat_c[i][self.feature_dict[key+'='+w]] = 1.
            # IDF
            if 'idf-file' in self.conf and self.idf is not None:
                for i, v in enumerate(ex['context']):
                    if v in self.idf:
                        feat_c[i][self.feature_dict['idf']] = self.idf[v]
                    else:
                        feat_c[i][self.feature_dict['idf']] = 0
            else:
                if 'idf' in ex:
                    # Use the IDF values in the dataset
                    for i, v in enumerate(ex['idf']):
                        feat_c[i][self.feature_dict['idf']] = v

        return context, feat_c, question, feat_q, ex['label'], ex['qid']


def batchify(batch):
    """collation_fn for data-loader; merge a list of samples to form a batch"""
    batch_size = len(batch)
    max_doc_length = max([ex[0].size(0) for ex in batch])
    max_q_length = max([ex[2].size(0) for ex in batch])
    ft_c_size = batch[0][1].size(1) if batch[0][1] is not None else 0
    ft_q_size = batch[0][3].size(0)

    # x1 (context)
    x1 = torch.LongTensor(batch_size, max_doc_length).zero_()
    # x1_mask (mask tensor of the context in the fixed length)
    x1_mask = torch.ByteTensor(batch_size, max_doc_length).fill_(1)
    # x1_f (context feature vector)
    x1_f = None
    if ft_c_size > 0:
        x1_f = torch.FloatTensor(batch_size, max_doc_length, ft_c_size).zero_()
    # x2 (question)
    x2 = torch.LongTensor(batch_size, max_q_length).zero_()
    # x2_mask
    x2_mask = torch.ByteTensor(batch_size, max_q_length).fill_(1)
    # x2_f (question feature vector)
    x2_f = torch.FloatTensor(batch_size, ft_q_size).zero_()

    # copy values
    for i, ex in enumerate(batch):
        clen = ex[0].size(0)
        qlen = ex[2].size(0)
        x1[i, :clen].copy_(ex[0])
        x1_mask[i, :clen].fill_(0)
        if ft_c_size > 0:
            x1_f[i, :clen].copy_(ex[1])
        x2[i, :qlen].copy_(ex[2])
        x2_mask[i, :qlen].fill_(0)
        x2_f[i, :ft_q_size].copy_(ex[3])
    labels = torch.LongTensor([ex[4] for ex in batch])
    qids = [ex[5] for ex in batch]

    return x1, x1_f, x1_mask, x2, x2_f, x2_mask, labels, qids


# ------------------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------------------

def load_data(datafile):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    return [json.loads(line.rstrip()) for line in open(datafile)]


def build_feature_dict(examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}
    # Part of speech tag features
    # The feature lists in a new mode have two sequences: title and abstract
    for ex in examples:
        for w in ex['pos']:
            _insert('pos=%s' % w)

    # Named entity tag features
    for ex in examples:
        for w in ex['ner']:
            _insert('ner=%s' % w)

    # idf value
    if 'idf' in examples[0]:
        _insert('idf')

    return feature_dict


def build_word_dict(examples):
    """Return a dictionary from provided examples. """
    word_dict = Dictionary()
    for w in load_words(examples):
        word_dict.add(w)
    return word_dict


def load_words(examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            words.add(w)
    words = set()
    qids = set()
    for ex in examples:
        if ex['qid'] not in qids:
            qids.add(ex['qid'])
            _insert(ex['question'])
        _insert(ex['context'])
    return words


# ------------------------------------------------------------------------------
# Utils
# - torch_summarize: displays the summary note with weights and parameters of
#  the network (obtained from http://bit.ly/2glYWVV)
# ------------------------------------------------------------------------------

def torch_summarize(model, show_weights=False, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights
    author: wassname
    url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
    license: MIT
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model.network._modules.items():
        # if it contains layers let call it recursively to get params and weights
        is_container = type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
            torch.nn.Module
        ]
        if is_container:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'
        if is_container:
            tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
