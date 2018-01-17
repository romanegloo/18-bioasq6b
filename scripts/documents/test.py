#!/usr/bin/env python3
"""Test script: given the year, for example '4' for Task 4B, it runs full
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

from BioAsq6B import retriever, reranker
from BioAsq6B.common import Timer, AverageMeter, measure_performance

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, default=4, choices=[1,2,3,4,5,6],
                    help='The year of the data to test on')
parser.add_argument('--query-model', type=str, default='sdm',
                    help='document retrieval model')
parser.add_argument('--rerank', action='store_true',
                    help='Enable re-ranker using qa_proximity model')
parser.add_argument('--qaprox-model', type=str,
                    help='Path to a QA_Proximity model')
parser.add_argument('--ndocs', type=int, default=10,
                    help='Number of document to retrieve')
parser.add_argument('--score-lambda', type=float, default=None,
                    help='Weight on retrieval scores')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose mode')
args = parser.parse_args()

# set defaults
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')
args.test_dir = os.path.join(DATA_DIR, 'bioasq/test')
args.index_path = os.path.join(DATA_DIR, 'galago-medline-full-idx')
args.database = os.path.join(DATA_DIR, 'concepts.db')
if args.qaprox_model is None:
    args.qaprox_model = os.path.join(DATA_DIR, 'qa_prox/var/best.mdl')

# Retriever
print('initializing retriever...')
doc_ranker = retriever.get_class('galago')(args)
# Re-ranker
if args.rerank:
    print('initializing re-ranker...')
    re_ranker = reranker.RerankQaProx(args)


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
    with open(data_file) as f:
        data = json.load(f)

    # Set tabular format for eval results
    table = prettytable.PrettyTable(['Question', 'GT',
                                     'Returned', 'Scores'])
    table.align['Question'] = 'l'
    table.align['Returned'] = 'r'
    table.max_width['Question'] = 40

    avg_prec, avg_recall, avg_f1, map, logp = \
        (AverageMeter() for i in range(5))
    for i, q in enumerate(data['questions']):
        if q['id'] not in qlist:
            continue
        prec, recall, f1, ap = _run(q, table)
        print('[seq #{} / {}]'.format(i+1, len(data['questions'])))
        print('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
              'avg_precision: {:.4f}'.format(prec, recall, f1, ap))
        avg_prec.update(prec)
        avg_recall.update(recall)
        avg_f1.update(f1)
        map.update(ap)
        logp.update(math.log(prec + 1e-6))
        sys.stdout.flush()

    gmap = math.exp(logp.avg)
    report = 'mean_precision: %.4f, mean_recall: %.4f, mean_f1: %.4f\n' % \
              (avg_prec.avg, avg_recall.avg, avg_f1.avg)
    report += 'map: {:.4f}, gmap: {:.4f}\n'.format(map.avg, gmap)
    print(report)


def _run(q, table):
    (docids, ret_scores) = doc_ranker.closest_docs(q, k=args.ndocs)
    if args.rerank:
        print('Re-ranking the results...')
        rel_scores = re_ranker.get_prox_scores(docids, q)

        # Compute final scores; merge_scores returns list of OrderedDict
        results = re_ranker.merge_scores(docids, ret_scores, rel_scores)
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


print('COMMAND: %s' % ' '.join(sys.argv))
if args.year <= 4:
    run_by_batches()
else:
    run_by_qlist()
