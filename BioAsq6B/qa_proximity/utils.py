#!/usr/bin/env python3
"""utilities (and data structure) for qa_proximity model"""
import logging
import json
import unicodedata
import os

import torch
from torch.nn.modules.module import _addindent
from torch.utils.data import Dataset
import numpy as np

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
    def __init__(self, args, examples, word_dict, feature_dict):
        self.ex = examples
        self.word_dict = word_dict
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        ex = self.ex[idx]

        # Index words
        # todo. check this part. LongTensor then batchify?
        context = torch.LongTensor([self.word_dict[w] for w in ex['context']])
        question = torch.LongTensor([self.word_dict[w] for w in ex['question']])

        # Create feature vector
        feat_c = torch.zeros(len(ex['context']), len(self.feature_dict))
        feat_q = torch.zeros(len(ex['question']), len(self.feature_dict))

        # Feature POS
        for pos in ['pos', 'q_pos']:
            for i, w in enumerate(ex[pos]):
                if 'pos='+w in self.feature_dict:
                    if pos == 'pos':
                        feat_c[i][self.feature_dict['pos='+w]] = 1.0
                    else:
                        feat_q[i][self.feature_dict['pos='+w]] = 1.0

        # Feature NER
        for ner in ['ner', 'q_ner']:
            for i, w in enumerate(ex[ner]):
                if 'pos='+w in self.feature_dict:
                    if ner == 'ner':
                        feat_c[i][self.feature_dict['ner='+w]] = 1.0
                    else:
                        feat_q[i][self.feature_dict['ner='+w]] = 1.0

        return context, feat_c, question, feat_q, ex['label'], ex['qid']


def batchify(batch):
    """collation_fn for data-loader; merge a list of samples to form a batch"""
    contexts = [ex[0] for ex in batch]
    features_c = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    features_q = [ex[3] for ex in batch]
    y = [ex[4] for ex in batch]
    qids = [ex[5] for ex in batch]

    # Batch documents and features
    max_length = max([c.size(0) for c in contexts])
    x1 = torch.LongTensor(len(contexts), max_length).zero_()
    x1_mask = torch.ByteTensor(len(contexts), max_length).fill_(1)
    x1_f = torch.zeros(len(contexts), max_length, features_c[0].size(1))
    for i, c in enumerate(contexts):
        x1[i, :c.size(0)].copy_(c)
        x1_mask[i, :c.size(0)].fill_(0)
        x1_f[i, :c.size(0)].copy_(features_c[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    x2_f = torch.zeros(len(questions), max_length, features_q[0].size(1))
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
        x2_f[i, :q.size(0)].copy_(features_q[i])

    # Y
    y = torch.LongTensor(y)

    return x1, x1_f, x1_mask, x2, x2_f, x2_mask, y, qids


# ------------------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------------------

def load_data(data_dir, year=None):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    examples = []
    # read all of relevant and irrelevant data
    if year is None:
        files = ['rel.txt', 'irrel.txt']
    else:
        files = ['rel-t{}.txt'.format(year), 'irrel-t{}.txt'.format(year)]
    for file in files:
        with open(os.path.join(data_dir, file)) as f:
            examples.extend([json.loads(line) for line in f])
    return examples


def build_feature_dict(examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}
    # Part of speech tag features
    for ex in examples:
        for w in ex['pos']:
            _insert('pos=%s' % w)

    # Named entity tag features
    for ex in examples:
        for w in ex['ner']:
            _insert('ner=%s' % w)

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
    curr_qid = ''
    for ex in examples:
        if curr_qid != ex['qid']:
            curr_qid = ex['qid']
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
