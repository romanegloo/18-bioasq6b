#!/usr/bin/env python3
"""Model Architecture"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from tqdm import tqdm

from BioAsq6B import PATHS
from BioAsq6B.QaSimSent import utils
from BioAsq6B.QaSimSent.network import QaSimBiRNN

logger = logging.getLogger()
logging.getLogger("gensim.models.keyedvectors").setLevel(logging.ERROR)


class QaSimSent(object):
    def __init__(self, conf, word_dict, feature_dict=None, state_dict=None):
        # book-keeping
        self.conf = conf
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.conf['num-features'] = len(feature_dict) if feature_dict else 0
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.network = QaSimBiRNN(conf)
        self.optimizer = None
        # load saved state, if exists
        if state_dict:
            self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network."""
        for p in self.network.encoder.parameters():
            p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.conf['optimizer'] == 'sgd':
            logger.info('Optimizer: SGD (learning_rate: {} '
                        'momentum: {}, weight_decay: {})'
                        ''.format(self.conf['learning-rate'],
                                  self.conf['momentum'],
                                  self.conf['weight-decay']))
            self.optimizer = \
                optim.SGD(parameters, self.conf['learning_rate'],
                          momentum=self.conf['momentum'],
                          weight_decay=self.conf['weight-decay'])
        elif self.conf['optimizer'] == 'adamax':
            logger.info('Optimizer: Adamax (weight-decay: {})'
                        ''.format(self.conf['weight-decay']))
            self.optimizer = \
                optim.Adamax(parameters, weight_decay=self.conf['weight-decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.conf['optimizer'])

    def load_embeddings(self, embedding_file=None, words=None):
        """Load pre-trained embeddings for a given list of words; assume that
        the file is in word2vec binary format"""
        if words is not None:
            raise RuntimeError("Training is done in a separate project: "
                               "refer to qasim_sent")
        if embedding_file is None:
            embedding_file = PATHS['embedding_file']
        # Read word vectors
        with open(embedding_file) as f:
            logger.info('Reading a word embedding file ({})...'
                        ''.format(embedding_file))
            size, dim = map(int, f.readline().split())
            assert dim == self.conf['embedding-dim']
            logger.info('Initializing space for embeddings...')
            # Replace embedding data; Add two for UNK
            self.network.encoder = nn.Embedding(size+1, dim, padding_idx=0)
            self.network.encoder.weight.requires_grad = False
            emb_data = self.network.encoder.weight.data
            # Replace word_dict
            self.word_dict = utils.Dictionary()
            pbar = tqdm(total=size)
            for line in f:
                v = line.split(' ')
                try:
                    self.word_dict.add(v[0])
                    emb_data[self.word_dict[v[0]]] = \
                        torch.FloatTensor([float(s) for s in v[1:dim+1]])
                except ValueError:
                    continue
                pbar.update()
            pbar.close()
        logger.info('Copied {} word embeddings'.format(len(emb_data)-2))

    def update_embeddings(self, words, embedding_file):
        """Updating this models embeddings with unseen words for later use"""
        logger.warning("update_word_embeddings: this part needs a fix")
        # embedding = self.network.encoder.weight.data
        # w2v_model = KeyedVectors.load_word2vec_format(embedding_file,
        #                                               binary=True)
        # cnt = 0
        # for w in words:
        #     if w in w2v_model:
        #         cnt += 1
        #         vec = torch.from_numpy(w2v_model[w]).float().unsqueeze(0)
        #         embedding = torch.cat((embedding, vec))
        #         self.word_dict[w] = len(self.word_dict)
        # logger.info('Added {} embeddings ({:.2f}%)'
        #             ''.format(cnt, 100 * cnt / len(words)))
        # self.network.encoder.weight.data = embedding
        # self.conf['vocab-size'] = len(self.word_dict)

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights
        ex components:
            x1, x1_f, x1_mask, x2, x2_f, x2_mask, labels, qids
        """
        # Train mode
        self.network.train()

        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:6]]
            target = Variable(ex[6].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:6]]
            target = Variable(ex[6])

        # Run forward
        scores = self.network(*inputs)
        loss = F.binary_cross_entropy(F.sigmoid(scores), target.float())

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.conf['grad-clipping'])

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
        ex_ = ex if len(ex) == 7 else ex[:7]

        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex_]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex_]
        # Forward
        scores = self.network(*inputs)
        return scores

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

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'conf': self.conf,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename):
        logger.info('Loading QA_Prox model {}'.format(filename))
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch'] if 'epoch' in saved_params else 0
        conf = saved_params['conf']
        return QaSimSent(conf, word_dict, feature_dict, state_dict=state_dict),\
               epoch