#!/usr/bin/env python3
"""Model Architecture"""

import logging
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.args.num_features = len(feature_dict)
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

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights
        ex components:
            x1, x1_f, x1_mask, x2, x2_f, x2_mask, y, qids
        """
        # Train mode
        self.network.train()

        # Add process for transferring data to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:6]]
            target = Variable(ex[6].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:6]]
            target = Variable(ex[6])

        # Run forward
        scores = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.cross_entropy(scores, target)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # todo. maybe add clip gradients here

        # Update weights
        self.optimizer.step()
        self.updates += 1

        return loss.data[0], ex[0].size(0)


    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:6]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:6]]

        # Forward
        scores = self.network(*inputs)
        return scores.max(1)[1]

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)