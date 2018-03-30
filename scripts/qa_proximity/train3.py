"""Training the BioAsq QA_Sim (span-based) Model"""
import logging
import time
import uuid
from pathlib import Path
from pprint import pformat
from collections import OrderedDict
import itertools
import re
import torch
from torch.utils.data import DataLoader, sampler

from BioAsq6B import common
from BioAsq6B.qa_proximity import utils, QaSimSpan

logger = logging.getLogger()


def init():
    # Initialize logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Setting runtime environment
    set_defaults()


def set_defaults():
    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------
    conf['debug'] = True
    # Identifiable name for each run
    conf['run-name'] = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    # Year of test; Datafiles for specific year is used
    conf['year'] = None
    # Random seed for all numpy/torch/cuda operations (for reproducibility)
    conf['random-seed'] = 12345
    # Train on CPU, even if GPUs are available
    conf['no-cuda'] = False
    # Specify GPU device id to use
    conf['gpu'] = 1
    # How many subprocesses to use for data loading.
    conf['num-workers'] = 0
    # Use DataParallel on all available GPUs
    conf['parallel'] = False
    # Train data iterations
    conf['num-epochs'] = 100
    # Print out model parameters
    conf['print-parameters'] = True
    # Save model + optimizer state after each epoch
    conf['checkpoint'] = True
    # Path to a pretrained model
    conf['pretrained'] = None

    # --------------------------------------------------------------------------
    # Paths (directories in Path object, files in string path)
    # --------------------------------------------------------------------------
    # Path to the directory containing test/train files
    conf['data-dir'] = Path(__file__).absolute().parents[3] / 'data/qa_prox'
    # Path to var directory; log files stored
    conf['var-dir'] = conf['data-dir'] / 'var'
    # Path to space-separated embeddings file
    filename_ = 'embeddings/wikipedia-pubmed-and-PMC-w2v.bin'
    conf['embedding-file'] = (conf['data-dir'] / filename_).as_posix()
    # IDF dict file
    conf['idf-file'] = (conf['data-dir'] / 'idf.p').as_posix()
    if conf['debug']:
        conf['embedding-file'] = None
        conf['idf-file'] = None
    # Checkpoint File
    conf['checkpoint-file'] = \
        conf['var-dir'] / (conf['run-name'] + '.checkpoint')


    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Type of the RNN layer [gru, lstm]
    conf['rnn-type'] = 'lstm'
    # Word embedding dimension
    conf['embedding-dim'] = 200
    # GRU hidden dimension
    conf['hidden-size'] = 128
    # Dataloader batch size
    conf['batch-size'] = 128
    # Combine hidden states from each encoding layer
    conf['concat-rnn-layers'] = True
    # Number of RNN layers stacked
    conf['num-rnn-layers'] = 1
    # Use additional features to word embeddings
    conf['features'] = ['pos', 'ner', 'idf']
    # Question Types
    conf['question-types'] = ['yesno', 'factoid', 'list', 'summary']
    # Optimizer: sgd or adamax
    conf['optimizer'] = 'adamax'
    # Weight decay factor for optimizer
    conf['weight-decay'] = 1e-6
    # Gradient clipping
    conf['grad-clipping'] = 10
    # Learning rate for SGD only
    conf['learning-rate'] = 0.01
    # Momentum factor
    conf['momentum'] = 0.9

    # Derived
    conf['cuda'] = not conf['no-cuda'] and torch.cuda.is_available()


def add_file_logger():
    # remove old file handlers
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    # add file log handle (args.var_dir and args.run_name need to be defined)
    log_file = \
        (conf['var-dir'] / 'run{}.log'.format(conf['run-name'])).as_posix()
    file = logging.FileHandler(log_file)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file.setFormatter(fmt)
    logger.addHandler(file)


def model_setup(options):
    global stats, train_loader, dev_loader

    add_file_logger()  # Set a new file logger handle
    # Override configurations
    for k in options.keys():
        conf[k] = options[k]
    logger.info(pformat(OrderedDict(sorted(conf.items(), key=lambda t: t[0]))))
    stats = {'epoch': 0, 'best_valid': 0, 'best_valid_at': 0,
             'acc_train': [], 'acc_test': [], 'losses': [], 'config': options}

    # Initialize model
    from_scratch = True
    model_ = None
    if conf['checkpoint']:
        conf['checkpoint-file'] = \
            conf['var-dir'] / (conf['run-name'] + '.checkpoint')
        print(conf['checkpoint-file'])
        if conf['checkpoint-file'].is_file():
            logger.info('Continue training a model...')
            model_, epoch = QaSimSpan.load(conf['checkpoint-file'])
            stats['epoch'] = epoch
            train_loader, dev_loader, word_dict, feature_dict = \
                prepare_dataloader(model_.word_dict, model_.feature_dict)
            from_scratch = False
    if from_scratch:
        logger.info('Initializing a model from scratch')
        train_loader, dev_loader, word_dict, feature_dict = prepare_dataloader()
        model_ = QaSimSpan(conf, word_dict, feature_dict)

    if conf['print-parameters']:
        logger.info(utils.torch_summarize(model_))

    # Set CPU/GPU mode
    if conf['cuda']:
        torch.cuda.set_device(conf['gpu'])
        logger.info('CUDA enabled (GPU %d)' % torch.cuda.current_device())
        model_.cuda()
    else:
        logger.info('Running on CPU only.')

    # Use multiple GPUs?
    if conf['parallel']:
        model_.parallelize()

    # Load pretrained embeddings
    if conf['embedding-file']:
        model_.load_embeddings(word_dict.tokens(), conf['embedding-file'])

    # Set optimizer
    model_.init_optimizer()

    return model_, train_loader, dev_loader, stats


def prepare_dataloader(word_dict=None, feature_dict=None):
    """Create data loaders for train and dev"""
    # Load examples
    logger.info('-' * 100)
    logger.info('Loading Datasets...')
    if conf['debug']:
        file_ = 'examples_y6_toy.txt'
    else:
        file_ = 'examples_y6.txt'
    train_ex = utils.load_data(conf['data-dir'] / 'train' / file_)
    logger.info('{} train examples loaded'.format(len(train_ex)))
    test_ex = utils.load_data(conf['data-dir'] / 'dev' / file_)
    logger.info('{} test examples loaded'.format(len(test_ex)))

    # Prepare feature_dict, word_dict
    if feature_dict is None:
        if len(conf['features']) > 0:
            logger.info('Building feature dictionary...')
            feature_dict = utils.build_feature_dict(train_ex)
            logger.info('Num features = {}'.format(len(feature_dict)))
            logger.info(feature_dict)
    if word_dict is None:
        logger.info('Build word dictionary...')
        word_dict = utils.build_word_dict(train_ex + test_ex)
        logger.info('Num words = %d' % len(word_dict))
    conf['vocab-size'] = len(word_dict)

    # Prepare DataLoaders
    logger.info('-' * 100)
    logger.info('Creating DataLoaders')
    train_dataset = utils.QaProxDataset(conf, train_ex, word_dict, feature_dict)
    train_loader_ = DataLoader(
        train_dataset,
        batch_size=conf['batch-size'],
        sampler=sampler.RandomSampler(train_dataset),
        collate_fn=utils.batchify,
        num_workers=conf['num-workers'],
        pin_memory=conf['cuda']
    )
    dev_dataset = utils.QaProxDataset(conf, test_ex, word_dict, feature_dict)
    dev_loader_ = DataLoader(
        dev_dataset,
        batch_size=conf['batch-size'],
        sampler=sampler.RandomSampler(dev_dataset),
        collate_fn=utils.batchify,
        num_workers=conf['num-workers'],
        pin_memory=conf['cuda']
    )
    return train_loader_, dev_loader_, word_dict, feature_dict


def run_epoches():
    cnt_earlystop = 3
    for epoch in range(stats['epoch'], conf['num-epochs'] + stats['epoch'] + 1):
        stats['epoch'] = epoch
        try:
            train()
            validate()
            best_updated = validate(mode='dev')
            if best_updated:
                cnt_earlystop = 3
                logger.info('Best valid: {:.2f} (epoch {}, {} updates)'
                            ''.format(stats['best_valid'], stats['epoch'],
                                      model.updates))
                # Checkpoint
                if conf['checkpoint']:
                    # Delete previously stored best models and save a new one
                    for f in conf['var-dir'].iterdir():
                        if re.search(r'^{}.*mdl$'.format(conf['run-name']),
                                     f.as_posix()):
                            f.unlink()
                    mdl_file = (conf['var-dir'] / '{}-best-acc{:d}.mdl'
                                ''.format(conf['run-name'],
                                          round(stats['best_valid']*100)))
                    logger.info('Saving a best model: {}'.format(mdl_file))
                    model.save(mdl_file.as_posix(), stats['epoch'] + 1)
            else:
                if stats['acc_train'][-1] > stats['acc_train'][-2] and \
                        stats['acc_test'][-1] < stats['acc_test'][-2]:
                    cnt_earlystop -= 1
            logger.info("early stop count: {}".format(cnt_earlystop))
            if cnt_earlystop == 0:
                break
        except KeyboardInterrupt:
            logger.warning('Terminating training loop')
            if conf['checkpoint']:
                model.save(conf['checkpoint-file'], stats['epoch'] + 1)
            report()
            exit(1)


def train():
    logger.info('-' * 100)
    logger.info('Starting training/validation loop...')

    # Initialize meters and timers
    train_loss = common.AverageMeter()
    train_loss_total = common.AverageMeter()

    # Run one epoch
    for idx, ex in enumerate(train_loader):
        loss = model.update(ex)
        train_loss.update(*loss)
        train_loss_total.update(*loss)
        if idx % 10 == 0:
            logger.info('train: Epoch = {} | iter = {}/{} | loss = {:.2E} '
                        ''.format(stats['epoch'], idx, len(train_loader),
                                  train_loss.avg))
            train_loss.reset()
    stats['losses'].append(train_loss_total.avg)

    # Checkpoint
    if conf['checkpoint']:
        model.save(conf['checkpoint-file'], stats['epoch'] + 1)


def validate(mode='train'):
    """Run one full validation"""
    acc_s = common.AverageMeter()
    acc_e = common.AverageMeter()
    acc_m = common.AverageMeter()
    best_updated = False
    examples = 0
    data_loader = train_loader if mode == 'train' else dev_loader
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s, pred_e, pred_score = model.predict(ex)
        accuracies = evaluate_accuracies(pred_s, pred_e, ex[6])
        acc_s.update(accuracies[0], batch_size)
        acc_e.update(accuracies[1], batch_size)
        acc_m.update(accuracies[2], batch_size)

        if mode == 'dev' and examples == 0:
            print(pred_s[0], pred_e[0], pred_score[0])
            print(ex[6])
        # If getting train accuracies, limit no of examples to validate
        examples += batch_size
        if mode == 'train' and examples >= 2e3:
            break

    logger.info("{} validation: examples = {} | "
                "accuracies = {:.2f} / {:.2f} / {:.2f}"
                ''.format(mode, examples, acc_s.avg, acc_e.avg, acc_m.avg))
    if mode == 'train':
        stats['acc_train'].append((acc_s.avg, acc_e.avg, acc_m.avg))
    else:
        sum_accuracies = acc_m.avg + acc_s.avg + acc_e.avg
        if sum_accuracies >= stats['best_valid']:
            stats['best_valid'] = sum_accuracies
            stats['best_valid_at'] = stats['epoch']
            best_updated = True
        stats['acc_test'].append(acc_m.avg)
    return best_updated


def evaluate_accuracies(pred_s, pred_e, labels):
    """Compute the correctness of the predictions to the list of answer
    spans; it's correct if the start/end position matches exactly to one
    of the answer spans."""

    batch_size = len(labels)
    start = common.AverageMeter()
    end = common.AverageMeter()
    em = common.AverageMeter()
    for i in range(batch_size):
        starts_ = [span[0] for span in labels[i]]
        ends_ = [span[1] for span in labels[i]]
        # Start matches
        if pred_s[i] in starts_:
            start.update(1)
        else:
            start.update(0)
        # End matches
        if pred_e[i] in ends_:
            end.update(1)
        else:
            end.update(0)
        # Both start and end match
        if any([1 for _s, _e in zip(starts_, ends_)
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100


def report():
    logger.info('-' * 100)
    logger.info('Report - RUN: {}'.format(conf['run-name']))
    logger.info('Best Valid: {} (epoch {})'
                ''.format(stats['best_valid'], stats['best_valid_at']))
    logger.info('Config: {}'.format(stats['config']))
    logger.info('Best Valid at: {:.2f}\t{}'.format(
        float(stats['best_valid'])*100, stats['best_valid_at']))


def dict_product(options):
    return (dict(zip(options, x)) for x in itertools.product(*options.values()))


if __name__ == '__main__':
    # Configure the Environment
    conf = dict()
    init()

    parameters = {
        'debug': [False],
        'rnn-type': ['lstm'],
        'batch-size': [32],
        'hidden-size': [256],
        'checkpoint': [True],
        'run-name': ['20180319-c988ff90']
    }
    for cmb in dict_product(parameters):
        model, train_loader, dev_loader, stats = model_setup(cmb)
        run_epoches()
        report()
