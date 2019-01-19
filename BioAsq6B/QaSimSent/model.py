#!/usr/bin/env python3
"""Model Architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import logging
import numpy as np
from tqdm import tqdm

from BioAsq6B import PATHS
from BioAsq6B.QaSimSent import utils
from BioAsq6B.QaSimSent.network import QaSimBiRNN

logger = logging.getLogger()


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

    def load_embeddings(self, embedding_file=None):
        """Load pre-trained embeddings for a given list of words; assume that
        the file is in word2vec binary format

        Args:
            embedding_file: path to an embedding file; either a binary format
                word embeddings or space separated text format embeddings
        """

        logger.info('Loading pre-trained word embeddings...')
        if embedding_file is None:
            embedding_file = PATHS['embedding_file']
        embedding = self.network.encoder.weight.data  # Embedding layer params
        update_word_dict = False
        # Read word vectors; It can be either space separated text file or C
        # binary format pre-trained embeddings.
        filename, ext = os.path.splitext(embedding_file)
        basename = os.path.basename(embedding_file)
        bl_write_subset = not basename.startswith('subset')
        subset_fh = None
        if bl_write_subset:
            subset_file = \
                os.path.join(os.path.dirname(embedding_file),
                             'subset-' + os.path.basename(embedding_file))
            logger.info('Writing a subset embedding file to [{}]'
                        ''.format(subset_file))
            subset_fh = open(subset_file, 'w')
        # If it's test mode, you get the full vocab of the given embeddings
        # If it's a training model for QASim, then you get the word_dict
        # from the training datasets.
        if ext == '.text':
            logger.info('Using gensim KeyedVectors, Reading word embedding '
                        'from [{}]...'.format(embedding_file))
            with open(embedding_file) as f:
                vocab_size, vec_dim = map(int, f.readline().split())
                assert vec_dim == self.conf['embedding-dim']
                if self.word_dict is None:
                    logger.info('Loading full vocabulary of the pre-trained '
                                'word embeddings for testing...')
                    self.network.encoder = nn.Embedding(vocab_size, vec_dim)
                    self.network.encoder.weight.requires_grad = False
                    embedding = self.network.encoder.weight.data
                    self.word_dict = utils.Dictionary()
                    update_word_dict = True
                pbar = tqdm(total=vocab_size)
                for line in f:
                    v = line.split(' ')
                    try:
                        if update_word_dict:
                            self.word_dict.add(v[0])
                        embedding[self.word_dict[v[0]]] = \
                            torch.FloatTensor([float(s) for s
                                               in v[1:vec_dim+1]])
                    except ValueError:
                        continue
                    pbar.update()
                pbar.close()
            logger.info('Copied {} word embeddings'.format(len(embedding)))
        elif ext == '.bin':
            logger.info('Using gensim KeyedVectors, Reading word embedding '
                        'from [{}]. It may take several minutes...'
                        ''.format(embedding_file))
            from gensim.models import KeyedVectors
            # Suppress gensim logging messages
            requests_logger = logging.getLogger('smart_open.smart_open_lib')
            requests_logger.setLevel(logging.INFO)

            wv_model = KeyedVectors.load_word2vec_format(embedding_file,
                                                         binary=True)
            vocab_size, vec_dim = (len(wv_model.vocab), wv_model.vector_size)
            if self.word_dict is None:
                logger.info('Loading full vocabulary of the pre-trained word '
                            'embeddings for testing...')
                self.network.encoder = nn.Embedding(vocab_size, vec_dim)
                self.network.encoder.weight.requires_grad = False
                embedding = self.network.encoder.weight.data
                self.word_dict = utils.Dictionary()
                map(self.word_dict.add, wv_model.vocab)
            for w in self.word_dict.tokens():
                try:
                    wv_ = wv_model.get_vector(w)
                except KeyError:
                    embedding[self.word_dict[w]].uniform_(-.25, 0.25)
                else:
                    embedding[self.word_dict[w]].copy_(torch.FloatTensor(wv_))
                    if bl_write_subset:
                        wv_str = ' '.join(np.char.mod('%f', wv_))
                        subset_fh.write('{} {}\n'.format(w, wv_str))
            logger.info('Copied {} embeddings'.format(len(embedding)-1))
        else:
            raise RuntimeError('Cannot determineThe word embedding filetype')

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
        if self.use_cuda:
            inputs = [e if e is None else e.cuda(async=True) for e in ex]
        else:
            inputs = [e if e is None else e for e in ex]
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
    def load(filename, load_wd=False):
        """
        :param filename: str: path to the saved model
        :param load_wd: bool: load the stored word_dict. If False, load full
        :return:
        """
        logger.info('Loading QASim model {}'.format(filename))
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict'] if load_wd else None
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch'] if 'epoch' in saved_params else 0
        conf = saved_params['conf']
        return QaSimSent(conf, word_dict, feature_dict, state_dict=state_dict),\
               epoch

