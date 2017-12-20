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

from BioAsq6B import retriever
from BioAsq6B.utils import Timer, AverageMeter, measure_performance

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pool', type=str, default='1,2,3',
                    help='comma separated list of years for training dataset; '
                         'If you test on 4th year, add 1,2,3 for training.')
parser.add_argument('-s', '--sample-size', type=float, default=.2,
                    help='sample of BioAsq training dataset;'
                         '< 1 percentage, >= 1 num. of samples, 0 all')
parser.add_argument('-q', '--qids', nargs='*',
                    help="evaluate on this question only")
parser.add_argument('--query-model', type=str, default='sdm',
                    help='document retrieval model')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose mode')
args = parser.parse_args()

# set defaults
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')
args.test_dir = os.path.join(DATA_DIR, 'bioasq/test')
args.index_path = os.path.join(DATA_DIR, 'galago-medline-full-idx')
args.database = os.path.join(DATA_DIR, 'concepts.db')

# set tabular format for eval results
table = prettytable.PrettyTable(['Question', 'GT', 'Returned'])
table.align['Question'] = 'l'
table.align['Returned'] = 'r'
table.max_width['Question'] = 40

# retriever
doc_ranker = retriever.get_class('galago')(args)

# read question/answer data
questions = []
print('reading training dataset...')
for year in args.pool.split(','):
    # read all batch files of the year
    for batch in range(1,6):
        if year == '1' and batch >= 4:
            continue
        batch_file = os.path.join(args.test_dir,
                                  "phaseB_{}b_0{}.json".format(year, batch))
        with open(batch_file) as f:
            data = json.load(f)
            if args.qids:
                questions.extend([q for q in data['questions']
                                 if q['id'] in args.qids])
            else:
                questions.extend(data['questions'])

# sampling
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
    questions = random.sample(questions, sample_size)

# RUN~
run_time = Timer()
avg_prec, avg_recall, avg_f1, map, logp = (AverageMeter() for i in range(5))

(lst_docid, lst_score) = doc_ranker.batch_closest_docs(questions, k=10)
for seq, res in enumerate(zip(lst_docid, lst_score, questions)):
    (docid, score, q) = res
    # read expected documents
    d_exp = []
    for line in q['documents']:
        m = re.match(".*pubmed/(\d+)$", line)
        if m:
            d_exp.append(m.group(1))
    table.clear_rows()
    q_fld = '[{}]\n{}'.format(q['id'], q['body'])
    table.add_row([q_fld, '\n'.join(d_exp),
                   '\n'.join(["{:8} [{:.4f}]".format(x[0], x[1])
                              for x in zip(docid, score)])])
    print(table)
    prec, recall, f1, ap = measure_performance(d_exp, docid)
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
