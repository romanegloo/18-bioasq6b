#!/usr/bin/env python3
"""(Note. This file is merged into run.py)
Train script: runs various retrieval methods for training purpose upon user
requests"""
import argparse
import os
import sys
import json
import prettytable
import random
import re
import math
import multiprocessing
from multiprocessing import Pool, Manager
from datetime import datetime
from termcolor import colored
import numpy as np
import logging
from pathlib import PosixPath
import pickle
from functools import partial
from collections import OrderedDict
# import traceback   # may need to use this to get the traceback generated
                     # from inside a thread or process

from BioAsq6B import retriever, reranker
from BioAsq6B.common import Timer, AverageMeter, measure_performance

logger = logging.getLogger()
doc_ranker = None
qasim_ranker = None
questions = []


def init():
    global doc_ranker, qasim_ranker, questions

    logger.setLevel(logging.DEBUG)
    # also use this logger in the multiprocessing module
    mlp = multiprocessing.log_to_stderr()
    mlp.setLevel(logging.WARN)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    if args.verbose:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
    else:
        file = logging.FileHandler(args.logfile)
        file.setFormatter(fmt)
        logger.addHandler(file)
        print("writing output in {}".format(args.logfile))
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Read cached QA_prox scores for faster evaluation
    cached_scores = None
    if args.cache_scores:
        manager = Manager()
        if os.path.isfile(args.score_datafile):
            # Confirm if a user wants to read the cached scores, or start
            # from the scratch
            ans = input("Cached proximity scores exist. Do you want to read "
                        "[Y/n]? ")
            if ans.lower().startswith('n'):
                cached_scores = manager.dict()
            else:
                print('Reading the scores: {}'.format(args.score_datafile))
                with open(args.score_datafile, 'rb') as f:
                    scores = pickle.load(f)
                cached_scores = manager.dict(scores)
        else:
            cached_scores = manager.dict()
    # Use cached galago retrieval results when running over the same set of
    #  question pools
    cached_retrievals = None
    if args.cache_retrieval:
        logger.info('cache_retrieval created')
        manager = Manager()
        cached_retrievals = manager.dict()

    logger.info('Initializing retriever...')
    doc_ranker = \
        retriever.get_class('galago')(args, cached_retrievals=cached_retrievals)
    if args.rerank:
        logger.info('Initializing re-ranker...')
        re_ranker = reranker.RerankQaSim(args, cached_scores=cached_scores)
        if args.print_parameters:
            from BioAsq6B.qa_proximity import utils
            model_summary = utils.torch_summarize(re_ranker.predictor.model)
            logger.info(model_summary)
        # check if the model needs to load idf data
        if re_ranker.predictor.model.args.use_idf:
            idf_file = os.path.join(DATA_DIR, 'qa_prox/idf.p')
            try:
                idf = pickle.load(open(idf_file, 'rb'))
            except:
                logger.error('Failed to read idf file from {}'.format(idf_file))
                raise
            logger.info('Using idf feature: {} loaded'.format(len(idf)))
            re_ranker.predictor.idf = idf

    # --------------------------------------------------------------------------
    # Read question/answer datasets
    # --------------------------------------------------------------------------
    if args.year == '5' or args.year == '6':
        batch_file = 'BioASQ-trainingDataset{}b.json'.format(args.year)
        batch_file = os.path.join(args.train_dir, batch_file)
        with open(batch_file) as f:
            data = json.load(f)
            if args.qids:
                questions.extend([q for q in data['questions']
                                  if q['id'] in args.qids])
            else:
                questions.extend(data['questions'])
    else:
        for year in args.year.split(','):
            # read all batch files of the year
            for batch in range(1, 6):
                if year == '1' and batch >= 4:
                    continue
                batch_file = "phaseB_{}b_0{}.json".format(year, batch)
                batch_file = os.path.join(args.test_dir, batch_file)
                logger.info('Reading train dataset from [{}]'
                            ''.format(batch_file))
                with open(batch_file) as f:
                    data = json.load(f)
                    if args.qids:
                        questions.extend([q for q in data['questions']
                                         if q['id'] in args.qids])
                    else:
                        questions.extend(data['questions'])


def sample_questions():
    if args.sample_size < 1:
        sample_size = int(len(questions) * args.sample_size)
        if args.sample_size == 0:
            sample_size = len(questions)
    else:
        sample_size = int(args.sample_size)
    if args.qids:  # overwrite sample_size
        sample_size = len(args.qids)

    if args.qids is None:
        if args.random_seed is not None:
            if args.random_seed == 0:
                random.seed()
            else:
                random.seed(args.random_seed)
        samples = random.sample(questions, sample_size)
    else:
        samples = questions
    logger.info('# of questions: {}, sample size: {}'
                ''.format(len(questions), sample_size))
    return samples


def query(q, seq=None):
    """run retrieval procedure (optionally rerank) of one question and return
    the result"""
    results = dict()
    results['question'] = [q['id'], q['body']]
    if seq:
        results['seq'] = seq
    (docids, ret_scores) = doc_ranker.closest_docs(q, k=args.ndocs)
    if args.rerank:
        rel_scores = qasim_ranker.get_prox_scores(docids, q)
        results['scores'] = qasim_ranker.merge_scores(args.score_weights, docids,
                                                      ret_scores, rel_scores)
    else:
        _scores = {docid: {'ret_score': score, 'score': score}
                   for docid, score in zip(docids, ret_scores)}
        results['scores'] = OrderedDict(sorted(_scores.items(),
                                               key=lambda t: t[1]['score'],
                                               reverse=True))
    # Read expected documents
    results['d_exp'] = []
    for line in q['documents']:
        m = re.match(".*pubmed/(\d+)$", line)
        if m:
            results['d_exp'].append(m.group(1))
    return results


def write_result(res, stats=None):
    """Display or write the results with performance measures"""
    # Set tabular format for eval results
    table = prettytable.PrettyTable(['Question', 'GT', 'Returned', 'Scores'])
    table.align['Question'] = 'l'
    table.align['Returned'] = 'r'
    table.max_width['Question'] = 40
    col0 = '[{}]\n{}'.format(res['question'][0], res['question'][1])
    col1 = '\n'.join(res['d_exp'])
    col2 = []  # returned documents
    col3 = []  # scores
    for k, v in res['scores'].items():
        k = colored(k, 'blue') if k in res['d_exp'] else k
        col2.append('{:>8}'.format(k))
        if 'rel_scores' in v:
            col3.append('({:.2f}, ({:.2f}/{:.2f}/{:.2f}), {:.4E})'
                        ''.format(v['ret_score'], v['rel_scores'][0],
                                  v['rel_scores'][1], v['rel_scores'][2],
                                  v['score']))
        else:
            col3.append('({:.4f}, {:.4E})'
                        ''.format(v['ret_score'], v['score']))
    col2 = '\n'.join(col2)
    col3 = '\n'.join(col3)
    table.add_row([col0, col1, col2, col3])
    prec, recall, f1, ap = \
        measure_performance(res['d_exp'], list(res['scores']), cutoff=10)
    report = ('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
             'avg_precision: {:.4f}').format(prec, recall, f1, ap)

    # Write out
    if stats is None or stats['epoch'] == 0:
        if args.verbose:
            print(table)
        else:
            with open(args.logfile, 'a') as f:
                f.write(table.get_string() + '\n')
                f.write(report + '\n')
        print('[#{}]'.format(res['question'][0]), report)
    else:
        logger.info('skip writing the retrieval results')

    # Update statistics
    if stats:
        stats['avg_prec'].update(prec)
        stats['avg_recall'].update(recall)
        stats['avg_f1'].update(f1)
        stats['map'].update(ap)
        stats['logp'].update(math.log(prec + 1e-6))
    return


def run(questions, epoch_no):
    # 'examine' mode
    if args.qids is not None:
        for q in questions:
            write_result(query(q))
        return

    run_time = Timer()
    stats = {
        'avg_prec': AverageMeter(),
        'avg_recall': AverageMeter(),
        'avg_f1': AverageMeter(),
        'map': AverageMeter(),
        'logp': AverageMeter(),
        'epoch': epoch_no
    }

    # Callback function to write the results of queries
    cb_write_results = partial(write_result, stats=stats)
    # Generate Pool for multiprocessing
    p = Pool(8)
    for seq, q in enumerate(questions):
        p.apply_async(query, args=(q, (seq, len(questions))),
                      callback=cb_write_results)
    p.close()
    p.join()

    # Report the overall batch performance measures
    gmap = math.exp(stats['logp'].avg)
    report = ("[batch #{} (run_time: {})]\n"
              "mean_precision: {:.4f}, mean_recall: {:.4f}, mean_f1: {:.4f} "
              "map: {:.4f}, gmap: {:.4f}"
              ).format(epoch_no, run_time.time(), stats['avg_prec'].avg,
                       stats['avg_recall'].avg, stats['avg_f1'].avg,
                       stats['map'].avg, gmap)
    with open(args.logfile, 'a') as f:
        f.write(report)
    print(report)

    # if caching score is enabled, store the scores
    if args.cache_scores:
        logger.info("Proximity Scores are saved")
        pickle.dump(dict(qasim_ranker.cached_scores),
                    open(args.score_datafile, 'wb'))

    # Returns 'map' score to validate the best
    return stats['map'].avg


def examine():
    """procedure for examining questions of interest"""
    samples = sample_questions()
    run(samples, 0)


def optimize():
    """best weights
        with weighted sum method:
            [0.16,0.10,0.73,0.79,0.21]
            [0.2,.3,0.5,0.7,0.3]
            [0.2,0.13,0.67,0.9,0.1]
        with rrf:
            [10,50,90,120]
            [50,90,100,100]
    """
    num_epoch = 100
    scores = []
    scores_sorted = None
    def _print_scores():
        # print best 5
        nonlocal scores_sorted
        report = ''
        scores_sorted = sorted(scores, key=lambda t: t[0], reverse=True)
        print("Current parameters: {}".format(args.score_weights))
        for v in scores_sorted[:5]:
            report += '\nBEST: [map: {:.4f}, params: {}]'.format(v[0], v[1])
        print(report)
        with open(args.logfile, 'a') as f:
            f.write(report + '\n')

    for i in range(num_epoch):
        # Sample the questions, if necessary
        samples = sample_questions()

        # Set hyperparameters; Override args.score_weights
        np.random.seed()
        if args.score_fusion == 'weighted_sum':  # weighted sum of the scores
            if args.score_weights:
                given_weights = list(map(float, args.score_weights.split(',')))
            else:
                # Just generate random numbers, which sums up to 1
                weights_ = np.random.random(4)
                given_weights = weights_.tolist()

            if i == 0:  # run with the given weights
                args.score_weights = \
                    ','.join(['{:.2f}'.format(w) for w in given_weights])
            else:
                best_weights = list(map(float, scores_sorted[0][1].split(',')))
                if i % 2 == 0:  # Generate new weights
                    weights_ = np.random.random(4)
                else:  # Change only one of the best weights
                    weights_ = best_weights
                    weights_[np.random.randint(0, 3)] = np.random.random()
                args.score_weights = ','.join(
                    ['{:.2f}'.format(w) for w in weights_])
        elif args.score_fusion == 'rrf':
            # reciprocal ranks fusion
            # : using flat scores of four modes (ret, rel_1, rel_2, rel_3)
            population = list(range(10, 210, 10))
            if args.score_weights:
                print(i, args.score_weights)
                given_weights = list(map(float, args.score_weights.split(',')))
            else:
                given_weights = np.random.choice(population, size=4,
                                                 replace=True)

            if i == 0:  # run with the given weights
                args.score_weights = \
                    ','.join(['{:.2f}'.format(w) for w in given_weights])
            else:
                best_weights = list(map(float, scores_sorted[0][1].split(',')))
                if i % 2 == 0:
                    weights_ = np.random.choice(population, size=4, replace=True)
                else:
                    weights_ = best_weights
                    weights_[np.random.randint(0, 3)] = \
                        np.random.choice(population)
                args.score_weights = ','.join(
                    ['{:.2f}'.format(w) for w in weights_])
        logger.info('Parameters: {}'.format(args.score_weights))

        # Run
        try:
            _map = run(samples, i)
            scores.append((_map, args.score_weights))
        except KeyboardInterrupt:
            _print_scores()
            raise
        _print_scores()

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Set Options
    # --------------------------------------------------------------------------
    """define parameters with the user provided arguments"""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-y', '--year', type=str, default='1,2,3,4',
                        help='Comma separated list of years for training'
                             ' dataset; If you test on 4th year, '
                             ' add 1,2,3 for training.')
    parser.add_argument('-s', '--sample-size', type=float, default=.2,
                        help='Sample of BioAsq training dataset;'
                             '< 1 percentage, >= 1 num. of samples, 0 all')
    parser.add_argument('-q', '--qids', nargs='*',
                        help="Evaluate on this question only")
    parser.add_argument('--query-model', type=str, default='sdm',
                        help='Document retrieval model')
    parser.add_argument('--rerank', action='store_true',
                        help='Enable re-ranker using qa_proximity model')
    parser.add_argument('-c', '--cache-scores', action='store_true',
                        help='Enable caching of the qa_prox scores')
    parser.add_argument('-r', '--cache-retrieval', action='store_true',
                        help='Enable caching galago retrieval results')
    parser.add_argument('--qaprox-model', type=str,
                        help='Path to a QA_Proximity model')
    parser.add_argument('--ndocs', type=int, default=10,
                        help='Number of document to retrieve')
    parser.add_argument('--score-weights', type=str, default=None,
                        help='comma separated weights of scoring functions')
    parser.add_argument('--score-fusion', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'rrf'],
                        help='Score fusion method')
    parser.add_argument('--random-seed', type=int, default=12345,
                        help='set a random seed for sampling operations')
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help='Filename to which retrieval results are saved')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose mode, print without writing on a log '
                             'file')
    parser.add_argument('--print-parameters', action='store_true',
                        help='Print out model parameters')

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

    args = parser.parse_args()

    # Set defaults
    args.run_id = datetime.today().strftime("%b%d-%H%M")
    DATA_DIR = os.path.join(
        PosixPath(__file__).absolute().parents[3].as_posix(), 'data')
    RUNS_DIR = os.path.join(
        PosixPath(__file__).absolute().parents[3].as_posix(), 'runs')
    args.test_dir = os.path.join(DATA_DIR, 'bioasq/test')
    args.train_dir = os.path.join(DATA_DIR, 'bioasq/train')
    args.index_path = os.path.join(DATA_DIR, 'galago-medline-full-idx')
    args.database = os.path.join(DATA_DIR, 'concepts.db')
    if args.qaprox_model is None:
        args.qaprox_model = os.path.join(DATA_DIR, 'qa_prox/var/best.mdl')
    if args.logfile is None:
        args.logfile = os.path.join(RUNS_DIR, args.run_id + ".log")
    else:
        args.logfile = os.path.join(RUNS_DIR, args.logfile)
    if args.cache_scores:
        args.score_datafile = \
            os.path.join(DATA_DIR,
                         'qa_prox/var/qa_scores-{}.pkl'.format(args.run_id))

    init()
    if args.qids is not None:
        examine()
    else:
        optimize()
