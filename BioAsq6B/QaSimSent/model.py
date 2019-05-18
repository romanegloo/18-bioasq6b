"""Model Architecture"""
import torch.optim as optim
import os
from network import *

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

    def load_embeddings(self, words, embedding_file):
        """Load pre-trained embeddings for a given list of words; assume that
        the file is in word2vec binary format"""
        words = {w for w in words if w in self.word_dict}
        embedding = self.network.encoder.weight.data
        # Read word vectors, which starts with a word token and space separated
        # numeric vector
        w2v_model = dict()
        with open(embedding_file) as f:
            logger.info('Reading a word embedding file ({})...'
                        ''.format(embedding_file))
            size, dim = map(int, f.readline().split())
            assert dim == self.conf['embedding-dim']
            for line in f:
                v = line.split(' ')
                try:
                    w2v_model[v[0]] = [float(s) for s in v[1:dim+1]]
                except ValueError:
                    continue
        logger.info('vocabulary size read: {}/{}, vec-dim: {}'
                    .format(len(w2v_model), size, dim))
        f_ = os.path.basename(embedding_file)
        if not (f_.startswith('subset') or f_.startswith('toy')) and \
                self.conf['hostname'] != 'job-instance':
            write_subset = True
        else:
            write_subset = False
        subset_lines = []
        for w in words:
            if w in w2v_model:
                if write_subset:
                    line = w + ' ' + ' '.join(map(str, w2v_model[w]))
                    subset_lines.append(line)
                embedding[self.word_dict[w]] = torch.FloatTensor(w2v_model[w])
        if write_subset:
            logger.info('Writing subset embedding file...')
            filename = os.path.join(os.path.dirname(embedding_file),
                                   'subset-' + os.path.basename(embedding_file))
            with open(filename, 'w') as f:
                f.write('{} {}\n'.format(size, dim))
                for line in subset_lines:
                    f.write(line + '\n')
        logger.info('Copied {} embeddings ({:.2f}%)'.format(
            len(embedding)-2, 100 * (len(embedding)-2) / len(words)))

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

    def predict(self, ex, top_n=1, max_len=None):
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

    def save(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'conf': self.conf,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename):
        logger.info('Loading QA_Prox model {}'.format(filename))
        saved_params = torch.load(filename,
                                   map_location=lambda storage, loc: storage)
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        conf = saved_params['conf']
        logger.info(conf)
        return QaSimSent(conf, word_dict, feature_dict, state_dict=state_dict), \
               saved_params['epoch']
