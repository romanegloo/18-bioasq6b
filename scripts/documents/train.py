#!/usr/bin/env python3
"""Train script: runs various retrieval methods for training purpose upon
user requests"""
import argparse
import os
import json
import prettytable
import random
import re
import math
from termcolor import colored

from BioAsq6B import retriever, reranker
from BioAsq6B.common import Timer, AverageMeter, measure_performance

# ------------------------------------------------------------------------------
# Set Options
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pool', type=str, default='1,2,3',
                    help='Comma separated list of years for training dataset; '
                         'If you test on 4th year, add 1,2,3 for training.')
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
parser.add_argument('--score-lambda', type=float, default=None,
                    help='Weight on retrieval scores')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Verbose mode')
args = parser.parse_args()

# Set defaults
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')
args.test_dir = os.path.join(DATA_DIR, 'bioasq/test')
args.index_path = os.path.join(DATA_DIR, 'galago-medline-full-idx')
args.database = os.path.join(DATA_DIR, 'concepts.db')
if args.qaprox_model is None:
    args.qaprox_model = os.path.join(DATA_DIR, 'qa_prox/var/best.mdl')

# Retriever
doc_ranker = retriever.get_class('galago')(args)
# Re-ranker
re_ranker = reranker.RerankQaProx(args)

# ------------------------------------------------------------------------------
# Read question/answer datasets
# ------------------------------------------------------------------------------
questions = []
for year in args.pool.split(','):
    # read all batch files of the year
    for batch in range(1,6):
        if year == '1' and batch >= 4:
            continue
        batch_file = os.path.join(args.test_dir,
                                  "phaseB_{}b_0{}.json".format(year, batch))
        print('reading train dataset from [{}]'.format(batch_file))
        with open(batch_file) as f:
            data = json.load(f)
            if args.qids:
                questions.extend([q for q in data['questions']
                                 if q['id'] in args.qids])
            else:
                questions.extend(data['questions'])

# Sampling
if args.sample_size < 1:
    sample_size = int(len(questions) * args.sample_size)
else:
    sample_size = int(args.sample_size)
if args.qids:  # overwrite sample_size
    sample_size = len(args.qids)
print("question size: %d, sample size: %d" % (len(questions), sample_size))

if args.qids:
    pass
else:
    random.seed()
    questions = random.sample(questions, sample_size)

# ------------------------------------------------------------------------------
# RUN~
# ------------------------------------------------------------------------------
run_time = Timer()
avg_prec, avg_recall, avg_f1, map, logp = (AverageMeter() for i in range(5))

# # Retrieve documents
# print('Retrieving documents...')
# (lst_docids, lst_ret_scores) = \
#     doc_ranker.batch_closest_docs(questions, k=args.ndocs)
#
# # Re-rank
# lst_rel_scores = None
# if args.rerank:
#     print('Re-ranking the results...')
#     lst_rel_scores = re_ranker.batch_get_prox_scores(lst_docids, questions)

# ------------------------------------------------------------------------------
# Display the results with performance measures
# ------------------------------------------------------------------------------
# Set tabular format for eval results
table = prettytable.PrettyTable(['Question', 'GT', 'Returned', 'Scores'])
table.align['Question'] = 'l'
table.align['Returned'] = 'r'
table.max_width['Question'] = 40

# if args.rerank:
#     answers = zip(questions, lst_docids, lst_ret_scores, lst_rel_scores)
# else:
#     answers = zip(questions, lst_docids, lst_ret_scores)

for seq, q in enumerate(questions):
    (docids, ret_scores) = doc_ranker.closest_docs(q)
    rel_scores = None
    if args.rerank:
       print('Re-ranking the results...')
       rel_scores = re_ranker.get_prox_scores(docids, q)

    # Compute final scores; merge_scores returns list of OrderedDict
    results = re_ranker.merge_scores(docids, ret_scores, rel_scores)

    # Read expected documents
    d_exp = []
    for line in q['documents']:
        m = re.match(".*pubmed/(\d+)$", line)
        if m:
            d_exp.append(m.group(1))
    # Print out
    table.clear_rows()
    col0 = '[{}]\n{}'.format(q['id'], q['body'])
    col1 = '\n'.join(d_exp)
    col2 = []  # returned documents
    col3 = []  # scores
    for k, v in results.items():
        k = colored(k, 'blue') if k in d_exp else k
        col2.append('{:>8}'.format(k))
        if 'rel_score' in v:
            col3.append('({:.4f}, {:.4f}, {:.4E})'
                        ''.format(v['ret_score'], v['rel_score'], v['score']))
        else:
            col3.append('({:.4f}, {:.4E})'.format(v['ret_score'], v['score']))
    col2 = '\n'.join(col2)
    col3 = '\n'.join(col3)
    table.add_row([col0, col1, col2, col3])
    print(table)
    prec, recall, f1, ap = measure_performance(d_exp, results.keys())
    print('batch #0: {}/{}'.format(seq+1, len(questions)))
    print('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
          'avg_precision: {:.4f}'.format(prec, recall, f1, ap))
    avg_prec.update(prec)
    avg_recall.update(recall)
    avg_f1.update(f1)
    map.update(ap)
    logp.update(math.log(prec + 1e-6))
gmap = math.exp(logp.avg)
report = '[batch #0 (run_time: {})]\n'.format(run_time.time())
report += 'mean_precision: %.4f, mean_recall: %.4f, mean_f1: %.4f' % \
          (avg_prec.avg, avg_recall.avg, avg_f1.avg)
report += '\nmap: {:.4f}, gmap: {:.4f}'.format(map.avg, gmap)
print(report)
