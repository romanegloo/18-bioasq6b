#!/usr/bin/env python3
"""Train script: runs various retrieval methods for training purpose upon
user requests"""
import argparse
import os
import sys
import json
import prettytable
import random
import re
import math
from multiprocessing import Pool
from datetime import datetime
from termcolor import colored
import numpy as np
import logging
from pathlib import PosixPath
# import traceback   # may need to use this to get the traceback generated
# from inside a thread or process

from BioAsq6B import retriever, reranker
from BioAsq6B.common import Timer, AverageMeter, measure_performance

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

doc_ranker = None
re_ranker = None
questions = []


def init():
    global doc_ranker, re_ranker, questions

    logger.info('Initializing retriever...')
    doc_ranker = retriever.get_class('galago')(args)
    if args.rerank:
        logger.info('Initializing re-ranker...')
        re_ranker = reranker.RerankQaProx(args)

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
    samples = []
    if args.sample_size < 1:
        sample_size = int(len(questions) * args.sample_size)
        if args.sample_size == 0:
            sample_size = len(questions)
    else:
        sample_size = int(args.sample_size)
    if args.qids:  # overwrite sample_size
        sample_size = len(args.qids)

    if args.qids:
        pass  # not implemented yet
    else:
        random.seed()
        samples = random.sample(questions, sample_size)
    logger.info('# of questions: {}, sample size: {}'
                ''.format(len(questions), sample_size))
    return samples


def _query(q):
    """run retrieval procedure (optionally rerank) of one question and return
    the result"""
    results = dict()
    results['question'] = [q['id'], q['body']]
    (docids, ret_scores) = doc_ranker.closest_docs(q, k=args.ndocs)
    if args.rerank:
        rel_scores = re_ranker.get_prox_scores(docids, q)
        results['scores'] = re_ranker.merge_scores(args.score_weights, docids,
                                                   ret_scores, rel_scores)
    else:
        results['scores'] = {docid: {'ret_score': score, 'score': score}
                             for docid, score in zip(docids, ret_scores)}

    # Read expected documents
    results['d_exp'] = []
    for line in q['documents']:
        m = re.match(".*pubmed/(\d+)$", line)
        if m:
            results['d_exp'].append(m.group(1))
    return results


def _write_result(res, fp=None, stats=None):
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
    if fp:
        fp.write(table.get_string() + '\n')
        fp.write(report + '\n')
        print('[#{}]'.format(res['question'][0]), report)
    else:
        print(table)
        print(report)
    if stats:
        stats['avg_prec'].update(prec)
        stats['avg_recall'].update(recall)
        stats['avg_f1'].update(f1)
        stats['map'].update(ap)
        stats['logp'].update(math.log(prec + 1e-6))
    return


def run(questions, epoch):
    run_time = Timer()
    stats = {
        'avg_prec': AverageMeter(),
        'avg_recall': AverageMeter(),
        'avg_f1': AverageMeter(),
        'map': AverageMeter(),
        'logp': AverageMeter()
    }

    # res_container = []
    def _log_results(res):
        """callback function for running _query()"""
        # res_container.append(res)
        with open(args.logfile, 'a') as f:
            _write_result(res, f, stats)

    p = Pool(10)
    for seq, q in enumerate(questions):
        p.apply_async(_query, args=(q,), callback=_log_results)
    p.close()
    p.join()

    gmap = math.exp(stats['logp'].avg)
    report = '[batch #{} (run_time: {})]\n'.format(epoch, run_time.time())
    report += 'mean_precision: %.4f, mean_recall: %.4f, mean_f1: %.4f' % \
              (stats['avg_prec'].avg, stats['avg_recall'].avg,
               stats['avg_f1'].avg)
    report += '\nmap: {:.4f}, gmap: {:.4f}'.format(stats['map'].avg, gmap)
    print(report)

    return stats['map'].avg


def optimize():
    num_epoch = 30
    scores = []
    for i in range(num_epoch):
        # Sample the questions, if necessary
        samples = sample_questions()

        # Set hyperparameters; Override args.score_weights
        rel_score_weights = np.random.uniform(low=0.1, high=0.9, size=3)
        rel_score_weights = \
            (rel_score_weights / rel_score_weights.sum()).tolist()
        hyper_weights = np.random.uniform(low=0.1, high=0.9, size=2)
        hyper_weights = (hyper_weights / hyper_weights.sum()).tolist()
        args.score_weights = ','.join(
            ['{:.2f}'.format(w) for w in (rel_score_weights + hyper_weights)])
        print('parameters: ', args.score_weights)

        # Run
        map = run(samples, i)
        scores.append((map, args.score_weights))

    # print best 5
    report = ''
    for i, v in sorted(enumerate(scores), key=lambda t: t[1][0], reverse=True):
        if i >= 5:
            break
        report += '\nBEST: [map: {:.4f}, params: {}]'.format(v[0], v[1])

    print(report)
    with open(args.logfile, 'a') as f:
        f.write(report)


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Set Options
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--qaprox-model', type=str,
                        help='Path to a QA_Proximity model')
    parser.add_argument('--ndocs', type=int, default=10,
                        help='Number of document to retrieve')
    parser.add_argument('--score-weights', type=str,
                        default='0.24,0.71,0.05,0.75,0.25',
                        help='Weights of scoring function;'
                             '[alpha,beta,gamma,lambda,mu]')
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help='Filename to which retrieval results are saved')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose mode')
    args = parser.parse_args()

    # Set defaults
    args.run_id = datetime.today().strftime("%b%d-%H%M")
    DATA_DIR = os.path.join(
        PosixPath(__file__).absolute().parents[3].as_posix(), 'data')
    # DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')
    RUNS_DIR = os.path.join(
        PosixPath(__file__).absolute().parents[3].as_posix(), 'runs')
    # RUNS_DIR = os.path.join(os.path.dirname(__file__), '../../../runs')
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

    init()
    optimize()
