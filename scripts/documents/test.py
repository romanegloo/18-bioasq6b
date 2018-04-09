#!/usr/bin/env python3
"""(Note. This file is merged into run.py)
Test script: given the year, for example '4' for Task 4B, it runs full
test on each batch of the year."""

import argparse
import os
import sys
import json
import prettytable
import re
import math
from termcolor import colored
from collections import OrderedDict
import numpy as np
import logging
from datetime import datetime
from pathlib import PosixPath
import multiprocessing
from multiprocessing import Pool
from functools import partial
import pickle

from BioAsq6B import retriever, reranker
from BioAsq6B.common import Timer, AverageMeter, measure_performance

logger = logging.getLogger()
doc_ranker = None
qasim_ranker = None
questions = []


def init():
    global doc_ranker, qasim_ranker, questions

    logger.setLevel(logging.INFO)
    # Also use this logger in the multiprocessing module
    mpl = multiprocessing.log_to_stderr()
    mpl.setLevel(logging.WARN)
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

    # Retriever
    logger.info('initializing retriever...')
    doc_ranker = retriever.get_class('galago')(args)
    # Re-ranker
    if args.rerank:
        logger.info('initializing re-ranker...')
        re_ranker = reranker.RerankQaSim(args)
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


def write_result(res, stats=None):
    """Write the results with performance measures"""
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
    if args.verbose:
        logger.info("Details:\n" + table.get_string() + '\n')
    logger.info('[#{}, {}/{}] {}'.format(res['question'][0], res['seq'][0],
                                         res['seq'][1], report))
    # Update statistics
    if stats:
        stats['avg_prec'].update(prec)
        stats['avg_recall'].update(recall)
        stats['avg_f1'].update(f1)
        stats['map'].update(ap)
        stats['logp'].update(math.log(prec + 1e-6))
    return


def run_by_batches():
    # Run by batches
    batch_report = ''
    for batch in range(1, 6):
        batch_time = Timer()
        avg_prec, avg_recall, avg_f1, map, logp = \
            (AverageMeter() for i in range(5))
        # read question list
        fields = ['id', 'body', 'documents']
        batch_file = os.path.join(args.test_dir,
                                  "phaseB_{}b_0{}.json".format(args.year,
                                                               batch))

        print('reading test dataset from [{}]'.format(batch_file))
        with open(batch_file) as f:
            data = json.load(f)

        # Set tabular format for eval results
        table = prettytable.PrettyTable(['Question', 'GT',
                                         'Returned', 'Scores'])
        table.align['Question'] = 'l'
        table.align['Returned'] = 'r'
        table.max_width['Question'] = 40

        for seq, q in enumerate(data['questions']):
            prec, recall, f1, ap = _run(q, table)
            print('batch #{}: {}/{}'.format(batch, seq+1,
                                            len(data['questions'])))
            print('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
                  'avg_precision: {:.4f}'.format(prec, recall, f1, ap))
            avg_prec.update(prec)
            avg_recall.update(recall)
            avg_f1.update(f1)
            map.update(ap)
            logp.update(math.log(prec + 1e-6))
            sys.stdout.flush()

        gmap = math.exp(logp.avg)
        report = '[batch #{} (run_time: {})]\n'.format(batch, batch_time.time())
        report += 'mean_precision: %.4f, mean_recall: %.4f, mean_f1: %.4f\n' %\
                  (avg_prec.avg, avg_recall.avg, avg_f1.avg)
        report += 'map: {:.4f}, gmap: {:.4f}\n'.format(map.avg, gmap)
        print(report)
        print('[{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]'
              ''.format(avg_prec.avg, avg_recall.avg, avg_f1.avg, map.avg,
                        gmap))
        batch_report += report
    print(batch_report)


def run_by_qlist():
    """only for year 5 and 6. It reads from training data for year 5 and 6
    instead of test batches. The results are assumed to be more accurate,
    since it has more hit documents (ground truth)."""
    # Read question list
    qlist_file = os.path.join(DATA_DIR,
                              'bioasq/train/qids_year{}.txt'
                              ''.format(args.year - 1))
    qlist = open(qlist_file).read().splitlines()
    data_file = os.path.join(DATA_DIR,
                             'bioasq/train/BioASQ-trainingDataset{}b.json'
                             ''.format(args.year))
    data = json.load(open(data_file))

    run_time = Timer()
    stats = {
        'avg_prec': AverageMeter(),
        'avg_recall': AverageMeter(),
        'avg_f1': AverageMeter(),
        'map': AverageMeter(),
        'logp': AverageMeter()
    }

    logger.info('running over {} questions'.format(len(data['questions'])))
    # Callback function to write the results of queries
    cb_write_results = partial(write_result, stats=stats)
    # Generate Pool for multiprocessing
    p = Pool(10)
    cnt = 0
    for q in data['questions']:
        if q['id'] not in qlist:
            continue
        p.apply_async(_query, args=(q, (cnt, len(qlist))),
                      callback=cb_write_results)
        cnt += 1
    p.close()
    p.join()

    # Report the overall batch performance measures
    gmap = math.exp(stats['logp'].avg)
    report = ("[Test Run {} (run_time: {})]\n"
              "mean_precision: {:.4f}, mean_recall: {:.4f}, mean_f1: {:.4f} "
              "map: {:.4f}, gmap: {:.4f}"
              ).format(args.run_id, run_time.time(), stats['avg_prec'].avg,
                       stats['avg_recall'].avg, stats['avg_f1'].avg,
                       stats['map'].avg, gmap)
    logger.info(report)


def _query(q, seq=None):
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

"""(deprecated)"""
def _run(q, table):
    (docids, ret_scores) = doc_ranker.closest_docs(q, k=args.ndocs)
    if args.rerank:
        rel_scores = qasim_ranker.get_prox_scores(docids, q)

        # Compute final scores; merge_scores returns list of OrderedDict
        results = qasim_ranker.merge_scores(args.score_weights, docids,
                                            ret_scores, rel_scores)
    else:
        ranked_list = {}
        ret_scores_norm = \
            np.exp(ret_scores) / np.sum(np.exp(ret_scores), axis=0)
        for i, d in enumerate(docids):
            ranked_list[d] = {'ret_score': ret_scores[i]}
            ranked_list[d]['score'] = ret_scores_norm[i]
        results = OrderedDict(sorted(ranked_list.items(),
                                     key=lambda t: t[1]['score'],
                                     reverse=True))

    # Read expected documents
    d_exp = []
    for line in q['documents']:
        m = re.match(".*pubmed/(\d+)$", line)
        if m:
            d_exp.append(m.group(1))
    # Print out
    table.clear_rows()
    if 'ideal_answer' in q:
        col0 = '[{}]\n{}\n=> {}'.format(q['id'], q['body'], q['ideal_answer'])
    else:
        col0 = '[{}]\n{}'.format(q['id'], q['body'])
    col1 = '\n'.join(d_exp)
    col2 = []  # returned documents
    col3 = []  # scores
    for k, v in results.items():
        k = colored(k, 'blue') if k in d_exp else k
        col2.append('{:>8}'.format(k))
        if 'rel_scores' in v:
            # ret_score, rel_(avg, median, max), total_score
            col3.append('({:.2f}, ({:.2f}/{:.2f}/{:.2f}), {:.4E})'
                        ''.format(v['ret_score'], v['rel_scores'][0],
                                  v['rel_scores'][1], v['rel_scores'][2],
                                  v['score']))
        else:
            col3.append('({:.4f}, {:.4E})'.format(v['ret_score'],
                                                  v['score']))
    col2 = '\n'.join(col2)
    col3 = '\n'.join(col3)
    table.add_row([col0, col1, col2, col3])
    print(table)
    return measure_performance(d_exp, list(results), 10)


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Set Options
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, default=6,
                        choices=[1,2,3,4,5,6],
                        help='The year of the data to test on')
    parser.add_argument('--query-model', type=str, default='sdm',
                        help='document retrieval model')
    parser.add_argument('--rerank', action='store_true',
                        help='Enable re-ranker using qa_proximity model')
    parser.add_argument('--qaprox-model', type=str,
                        help='Path to a QA_Proximity model')
    parser.add_argument('--ndocs', type=int, default=10,
                        help='Number of document to retrieve')
    parser.add_argument('--score-weights', type=str, default=None,
                        help='comma separated weights of scoring functions')
    parser.add_argument('--score-fusion', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'rrf'],
                        help='Score fusion method')
    parser.add_argument('-l', '--logfile', type=str, default=None,
                        help='Filename to which retrieval results are saved')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument('--print-parameters', action='store_true',
                         help='Print out model parameters')
    args = parser.parse_args()

    # set defaults
    args.run_id = datetime.today().strftime("%b%d-%H%M")
    DATA_DIR = os.path.join(
        PosixPath(__file__).absolute().parents[3].as_posix(), 'data')
    RUNS_DIR = os.path.join(
        PosixPath(__file__).absolute().parents[3].as_posix(), 'runs')
    args.test_dir = os.path.join(DATA_DIR, 'bioasq/test')
    args.index_path = os.path.join(DATA_DIR, 'galago-medline-full-idx')
    args.database = os.path.join(DATA_DIR, 'concepts.db')
    if args.qaprox_model is None:
        args.qaprox_model = os.path.join(DATA_DIR, 'qa_prox/var/best.mdl')
    if args.logfile is None:
        args.logfile = os.path.join(RUNS_DIR, args.run_id + ".log")
    else:
        args.logfile = os.path.join(RUNS_DIR, args.logfile)
    args.cache_retrieval = False
    args.cache_scores = False

    init()
    if args.year <= 4:
        run_by_batches()
    else:
        run_by_qlist()

