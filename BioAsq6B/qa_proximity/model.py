#!/usr/bin/env python3
"""Model Architecture"""

import logging
import torch
import torch.optim as optim
from gensim.models.keyedvectors import KeyedVectors

from .network import QaProxBiRNN

logger = logging.getLogger(__name__)


class QaProx(object):
    def __init__(self, args, state_dict=None, word_dict=None,
                 feature_dict=None):
        # book-keeping
        self.args = args
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        self.network = QaProxBiRNN(args)

        # load saved state, if exists
        if state_dict:
            self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network."""
        for p in self.network.encoder.parameters():
            p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    def load_embeddings(self, words, embedding_file):
        """Load pre-trained embeddings for a given list of words; assume that
        the file is in word2vec binary format"""
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.encoder.weight.data
        w2v_model = KeyedVectors.load_word2vec_format(embedding_file,
                                                      binary=True)
        for w in words:
            if w in w2v_model:
                vec = torch.from_numpy(w2v_model[w])
                # vec = torch.FloatTensor([float(i) for i in w2v_model[w]])
                embedding[self.word_dict[w]] = vec
        logger.info('Copied %d embeddings (%.2f%%)' %
                    (len(embedding), 100 * len(embedding) / len(words)))
        w2v_model = None

        # Below is for reading embedding file in text format, like FastText
        # ----------------------------------------------------------------------
        # vec_counts = {}
        # with open(embedding_file) as f:
        #     skip_first_line = False  # some formats starts with dimensions
        #     for line in f:
        #         if skip_first_line:
        #             skip_first_line = False
        #             continue
        #         parsed = line.rstrip().split(' ')
        #         assert(len(parsed) == embedding.size(1) + 1), line
        #         w = self.word_dict.normalize(parsed[0])
        #         if w in words:
        #             vec = torch.Tensor([float(i) for i in parsed[1:]])
        #             if w not in vec_counts:
        #                 vec_counts[w] = 1
        #                 embedding[self.word_dict[w]].copy_(vec)
        #             else:
        #                 logging.warning(
        #                     'WARN: Duplicate embedding found for %s' % w
        #                 )
        #                 vec_counts[w] = vec_counts[w] + 1
        #                 embedding[self.word_dict[w]].add_(vec)
        # for w, c in vec_counts.items():
        #     embedding[self.word_dict[w]].div_(c)
        # logger.info('Loaded %d embeddings (%.2f%%)' %
        #             (len(vec_counts), 100 * len(vec_counts) / len(words)))


