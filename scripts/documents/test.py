#!/usr/bin/env python3
"""Test script: given the year, for example '4' for Task 4B, it runs full
test on each batch of the year."""

import argparse
import os
import json
import prettytable
import re
import math

from BioAsq6B import retriever
from BioAsq6B.common import Timer, AverageMeter, measure_performance

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, default=4, choices=[1,2,3,4],
                    help='The year of the data to test on')
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

# run by batches
batch_report = ''
for batch in range(1, 6):
    batch_time = Timer()
    avg_prec, avg_recall, avg_f1, map, logp = (AverageMeter() for i in range(5))
    # read question list
    fields = ['id', 'body', 'documents']
    batch_file = os.path.join(args.test_dir,
                              "phaseB_{}b_0{}.json".format(args.year, batch))
    with open(batch_file) as f:
        data = json.load(f)

    # if running in batch, use this code----------------------------------------
    # print('running batch document search...#{}'.format(batch))
    # (lst_docid, lst_score) = doc_ranker.batch_closest_docs(data['questions'])
    # for seq, res in enumerate(zip(lst_docid, lst_score, data['questions'])):
    #     (docid, score, q) = res
    #---------------------------------------------------------------------------
    for seq, q in enumerate(data['questions']):
        (docid, score) = doc_ranker.closest_docs(q)
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
        print('batch #{}: {}/{}'.format(batch, seq+1, len(data['questions'])))
        print('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
              'avg_precision: {:.4f}'.format(prec, recall, f1, ap))
        avg_prec.update(prec)
        avg_recall.update(recall)
        avg_f1.update(f1)
        map.update(ap)
        logp.update(math.log(prec + 1e-6))

    gmap = math.exp(logp.avg)
    report = '[batch #{} (run_time: {})]\n'.format(batch, batch_time.time())
    report += 'mean_precision: %.4f, mean_recall: %.4f, mean_f1: %.4f\n' %\
              (avg_prec.avg, avg_recall.avg, avg_f1.avg)
    report += 'map: {:.4f}, gmap: {:.4f}\n'.format(map.avg, gmap)
    print(report)
    batch_report += report
print(batch_report)
