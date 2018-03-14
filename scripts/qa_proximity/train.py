#!/usr/bin/env python3
"""training a qa_proximity model"""

import argparse
import logging
import os
import sys
import re
import uuid
import time
import itertools
import pickle
import torch
from torch.utils.data import DataLoader, sampler

from BioAsq6B import common
from BioAsq6B.qa_proximity import utils, QaProx

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_arguments(parser):
    """define parameters with the user provided arguments"""
    parser.register('type', 'bool', str2bool)

    # Runtime Environment
    runtime = parser.add_argument_group('Runtime Environments')
    runtime.add_argument('--run-name', type=str,
                         help='Identifiable name for each run')
    runtime.add_argument('--random-seed', type=int, default=12345,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--no-cuda', action='store_true',
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help="Specify GPU device id to use")
    runtime.add_argument('--parallel', type=bool, default=False,
                         help="Use DataParallel on all available GPUs")
    runtime.add_argument('--num-epochs', type=int, default=30,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=256,
                         help='batch size for training')
    runtime.add_argument('--print-parameters', action='store_true',
                         help='Print out model parameters')
    runtime.add_argument('--save-plots', action='store_true',
                         help='Save plot files of losses/accuracies/etc.')

    # Files: set the paths to important files
    files = parser.add_argument_group('Files')
    files.add_argument('--data-dir', type=str,
                       help='Path to the directory containing test/train files')
    files.add_argument('--year', type=int, default=None,
                       help='Year of test; Datafiles for specific year is used')
    files.add_argument('--var-dir', type=str,
                       help='Path to var directory; log files stored')
    files.add_argument('--train-file', type=str,
                       help='Path to preprocessed train data file')
    files.add_argument('--test-file', type=str,
                       help='Path to preprocessed test data file')
    files.add_argument('--embedding-file', type=str,
                       help='path to space-separated embeddings file')

    # Model Architecture: model specific options
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--rnn-type', type=str, default='gru',
                       choices=['gru', 'lstm'], help='Type of the RNN')
    model.add_argument('--embedding-dim', type=int, default=200,
                       help='word embedding dimension')
    model.add_argument('--hidden-size', type=int, default=128,
                       help='GRU hidden dimension')
    model.add_argument('--concat-rnn-layers', type='bool', default=True,
                       help='Combine hidden states from each encoding layer')
    model.add_argument('--num-rnn-layers', type=int, default=1,
                       help='number of RNN layers stacked')
    model.add_argument('--uni-direction', action='store_true', default=False,
                       help='use single directional RNN')
    model.add_argument('--no-token-feature', action='store_true',
                       default=False, help='use only word embeddings')
    model.add_argument('--use-idf', action='store_true', default=False,
                       help='add inversed document frequency')

    # Optimization details
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd or adamax')
    model.add_argument('--weight-decay', type=float, default=1e-6,
                       help='Weight decay factor for optimizer')
    optim.add_argument('--dropout-emb', type=float, default=0,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')
    optim.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for SGD only')
    optim.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum factor')


    # Saving + Loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', action='store_true',
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')


def init():
    """set default values and initialize components"""
    # initialize logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logger.info('-' * 100)
    logger.info('Initializing...')

    # set default values
    args.class_num = 2  # relevant (0) or irrelevant (1)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.data_workers = 8

    # set paths
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__),
                                     '../../../data/qa_prox')
    if args.var_dir is None:
        args.var_dir = os.path.join(args.data_dir, 'var')
    if not os.path.exists(args.var_dir):
        os.mkdir(args.var_dir)

    if args.embedding_file is None:
        args.embedding_file = os.path.join(args.data_dir,
                                  'embeddings/wikipedia-pubmed-and-PMC-w2v.bin')
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set random state
    if args.random_seed is not None:
        if args.random_seed == 0:
            torch.manual_seed()
        else:
            torch.manual_seed(args.random_seed)
        # np.random.seed(args.random_seed)  # if numpy is used
        if args.cuda:
            if args.random_seed == 0:
                torch.cuda.manual_seed()
            else:
                torch.cuda.manual_seed(args.random_seed)


def add_file_logger():
    # remove old file handlers
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    # add file log handle (args.var_dir and args.run_name need to be defined)
    log_path = os.path.join(args.var_dir, 'run{}.log'.format(args.run_name))
    file = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file.setFormatter(fmt)
    logger.addHandler(file)


def prepare_dataloader():
    """Make data loaders for train and dev"""
    global args
    logger.info('-' * 100)
    logger.info('Loading Datasets...')
    train_ex = utils.load_data(os.path.join(args.data_dir, 'train'), args.year)
    logger.info('{} train examples loaded'.format(len(train_ex)))
    test_ex = utils.load_data(os.path.join(args.data_dir, 'test'), args.year)
    logger.info('{} test examples loaded'.format(len(test_ex)))

    feature_dict = None
    idf = None
    if not args.no_token_feature:
        logger.info('Building feature dictionary...')
        feature_dict = utils.build_feature_dict(train_ex)
        logger.info('Num features = {}'.format(len(feature_dict)))
        logger.info(feature_dict)

        if args.use_idf:
            idf = pickle.load(open(os.path.join(args.data_dir, 'idf.p'), 'rb'))
            logger.info('Using idf feature: {} loaded'.format(len(idf)))

    logger.info('Build word dictionary...')
    word_dict = utils.build_word_dict(train_ex + test_ex)
    logger.info('Num words = %d' % len(word_dict))
    args.vocab_size = len(word_dict)

    logger.info('-' * 100)
    logger.info('Creating DataLoaders')
    if args.cuda:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    else:
        kwargs = {'num_workers': args.data_workers}

    train_dataset = utils.QaProxDataset(args, train_ex, word_dict,
                                        feature_dict, idf)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler.RandomSampler(train_dataset),
        collate_fn=utils.batchify,
        **kwargs
    )
    test_dataset = utils.QaProxDataset(args, test_ex, word_dict,
                                       feature_dict, idf)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=sampler.RandomSampler(test_dataset),
        collate_fn=utils.batchify,
        num_workers=0
    )
    return train_loader, test_loader, word_dict, feature_dict


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------

def train(data_loader, model, global_stats):
    logger.info('-' * 100)
    logger.info('Starting training/validation loop...')

    # Initialize meters and timers
    train_loss = common.AverageMeter()
    train_loss_total = common.AverageMeter()
    epoch_time = common.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        loss = model.update(ex)[0]
        train_loss.update(loss)
        train_loss_total.update(loss)
        if idx % 10 == 0:
            logger.info('train: Epoch = {} | iter = {}/{} | loss = {:.2E} |'
                        ' Elapsed time = {:.2f}'
                        ''.format(global_stats['epoch'], idx, len(data_loader),
                                  train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
    global_stats['losses'].append(train_loss_total.avg)


def validate(data_loader, model, global_stats, mode):
    """Run one full validation"""
    epoch = global_stats['epoch']
    eval_time = common.Timer()
    acc = common.AverageMeter()
    best_updated = False

    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        scores = model.predict(ex)
        pred = scores.gt(0).long()
        acc_ = torch.LongTensor(ex[-2]).eq(pred.data.cpu()).sum() / batch_size
        acc.update(acc_, batch_size)

        # If getting train accuracies, limit no of examples to validate
        examples += batch_size
        if mode == 'train' and examples >= 2e3:
            break

    logger.info("{} validation: examples = {} | accuracy = {}"
                ''.format(mode, examples, acc.avg))
    if mode =='train':
        global_stats['acc_train'].append(acc.avg)
    else:
        if acc.avg > global_stats['best_valid']:
            global_stats['best_valid'] = acc.avg
            global_stats['best_valid_at'] = epoch
            best_updated = True
        global_stats['acc_test'].append(acc.avg)
    return best_updated


def report(stats):
    logger.info('-' * 100)
    logger.info('Report - RUN: {}'.format(args.run_name))
    logger.info('Best Valid: {} (epoch {})'
                ''.format(stats['best_valid'], stats['best_valid_at']))
    logger.info('Config: {}'.format(stats['config']))
    logger.info('Best Valid at: {:.2f}\t{}'.format(
        float(stats['best_valid'])*100, stats['best_valid_at']))
    if args.save_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = list(range(args.num_epochs))
        # losses
        fig = plt.figure(figsize=(9, 4))
        cnn = fig.add_subplot(121)
        cnn.plot(x, stats['losses'], 'r', label='train loss')
        cnn.set_xlabel('epoch')
        cnn.set_ylabel('loss')
        cnn.set_title('Train Losses')
        cnn.legend()

        # accuracy
        rnn = fig.add_subplot(122)
        rnn.plot(x, stats['acc_train'], 'g', label='train')
        rnn.plot(x, stats['acc_test'], 'r', label='test')
        rnn.set_ylim(ymax=1)
        rnn.set_xlabel('epoch')
        rnn.set_ylabel('accuracy')
        rnn.set_title('Test/Train Accuracies')
        rnn.legend(loc=4)

        plt.tight_layout()
        out_file = os.path.join(args.var_dir,
                                'plot-{}.png'.format(args.run_name))
        plt.savefig(out_file)


if __name__ == '__main__':
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    # --------------------------------------------------------------------------
    # Arguments
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        'package_name',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Initialization and DataLoaders
    # --------------------------------------------------------------------------
    init()

    # --------------------------------------------------------------------------
    # Train/Validation loop
    # --------------------------------------------------------------------------

    # configurables = [['g', 'l'], ['u', 'b'], [64], [100], [5, 6]]
    # configurables = [['g'], ['b'], [64],
    #                  [h for h in range(320, 420, 20)], [6]]
    configures = [('g', 'b', 64, 400, 6)]
    # for cfg in itertools.product(*configurables):
    for cfg in configures:
        args.run_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
        logger.info('RUN: {}'.format(args.run_name))
        add_file_logger()  # set a new logger file handle
        cnt_earlystop = 3
        args.rnn_type = 'gru' if cfg[0] == 'g' else 'lstm'
        args.uni_direction = True if cfg[1] == 'u' else False
        args.batch_size = cfg[2]
        args.hidden_size = cfg[3]
        args.year = cfg[4]
        stats = {'epoch': 0, 'timer': common.Timer(),
                 'best_valid': 0, 'best_valid_at': 0,
                 'acc_train': [], 'acc_test': [], 'losses': [], 'config': cfg}
        train_loader, test_loader, word_dict, feature_dict = prepare_dataloader()
        logger.info('Starting with Config: {}'.format(cfg))

        # ----------------------------------------------------------------------
        # Model setup
        # ----------------------------------------------------------------------
        model = QaProx(args, word_dict, feature_dict)
        model_summary = utils.torch_summarize(model)
        if args.print_parameters:
            logger.info(model_summary)

        # set cpu/gpu mode
        if args.cuda:
            torch.cuda.set_device(args.gpu)
            logger.info('CUDA enabled (GPU %d)' % torch.cuda.current_device())
            model.cuda()
        else:
            logger.info('Running on CPU only.')
        # Use multiple GPUs?
        if args.parallel:
            model.parallelize()

        # args.embedding_file = None  # for debugging
        if args.embedding_file:
            model.load_embeddings(word_dict.tokens(), args.embedding_file)
        # Set up optimizer
        model.init_optimizer()

        # ----------------------------------------------------------------------
        # Run training
        # ----------------------------------------------------------------------
        for epoch in range(0, args.num_epochs):
            stats['epoch'] = epoch
            try:
                train(train_loader, model, stats)
                validate(train_loader, model, stats, mode='train')
                best_updated = \
                    validate(test_loader, model, stats, mode='test')
                if best_updated:
                    cnt_earlystop = 3
                    logger.info('Best valid: {:.2f} (epoch {}, {} updates)'
                                ''.format(stats['best_valid'], stats['epoch'],
                                          model.updates))
                    # delete previous best models and save the best model
                    for f in os.listdir(args.var_dir):
                        if re.search(r'^{}.*mdl$'.format(args.run_name), f):
                            os.remove(os.path.join(args.var_dir, f))
                    model_file = \
                        os.path.join(args.var_dir,
                                     '{}-best-acc{:d}.mdl'
                                     ''.format(args.run_name,
                                               round(stats['best_valid']*100)))
                    model.save(model_file)
                else:
                    if stats['acc_train'][-1] > stats['acc_train'][-2] and \
                            stats['acc_test'][-1] < stats['acc_test'][-2]:
                        cnt_earlystop -= 1
                logger.info("early stop count: {}".format(cnt_earlystop))
                if cnt_earlystop <= 0:
                    break
            except KeyboardInterrupt:
                logger.warning('Training loop terminated')
                report(stats)
                exit(1)

        # --------------------------------------------------------------------------
        # Report the results
        # --------------------------------------------------------------------------
        report(stats)
        logger.info('COMMAND: %s' % ' '.join(sys.argv))
