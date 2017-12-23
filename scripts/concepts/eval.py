#!/usr/bin/env python3
"""evaluating "concepts" retrieval performance

: Relevance judgement data from "BioASQ5-training dataset". It consists of 1,
799 questions and its expected answers.
"""
import argparse
import json
import logging
import os
import random
import re
import sqlite3
from pathlib import PosixPath
import prettytable
import math

import common as utils

DATA_DIR = \
    os.path.join(PosixPath(__file__).absolute().parents[4].as_posix(), 'data')

# initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def evaluate(args):
    """
    :param args: expecting arguments of {dataset, sample_size, qid}
    :return: displays metrics
    """
    logger.info('Sampling relevance judgement file: %s' %
                os.path.basename(args.dataset))
    fields = ['id', 'body', 'concepts']
    sample_data = []
    with open(args.dataset) as f:
        data = json.load(f)
        data_size = len(data['questions'])
        if args.sample_size < 1:
            sample_size = int(data_size * args.sample_size)
        else:
            sample_size = int(args.sample_size)
        logger.info("question size: %d, sample size: %d" %
                    (data_size, sample_size))

        if args.qids is not None:
            for sample in data['questions']:
                if sample['id'] not in args.qids:
                    continue
                else:
                    question = {k: sample[k] for k in fields}
                    sample_data.append(question)
        else:
            for sample in random.sample(data['questions'], sample_size):
                try:
                    question = {k: sample[k] for k in fields}
                except KeyError as e:
                    print("keyerror: {}".format(sample['id']))
                    continue
                sample_data.append(question)

    # results table format
    table = prettytable.PrettyTable(
        ['Question', 'MeSH', 'GO', 'uniprot', 'DO', 'Returned']
    )
    table.align['Question'] = 'l'
    table.max_width['Question'] = 40

    # connect to concepts database
    cnx = sqlite3.connect(args.database)
    csr = cnx.cursor()
    sql = """SELECT * FROM concepts WHERE id=?;"""

    # evaluate!!
    precisions = []
    recalls = []
    f_scores = []
    avg_precisions = []
    for q in sample_data:
        logger.info('qid: {}'.format(q['id']))
        c_exp = {'mesh': [], 'go': [], 'uniprot': [], 'do': []}
        c_pred = {'mesh': set(), 'go': set(), 'uniprot': set(), 'do': set()}
        for line in q['concepts']:
            # MeSH
            m = re.match(".*mesh.*term=([a-zA-Z]+\d+)$", line)
            if m:
                c_exp['mesh'].append(m.group(1))
            # GO
            m = re.match(".*geneontology.*term=(GO:\d+)$", line)
            if m:
                c_exp['go'].append(m.group(1))
            # uniprot
            m = re.match(".*uniprot/(\w*)$", line)
            if m:
                c_exp['uniprot'].append(m.group(1))
            # Disease Ontology
            m = re.match(".*disease-ontology.*(DOID:\d+)$", line)
            if m:
                c_exp['do'].append(m.group(1))

        # read concepts from concepts.db
        rows = csr.execute(sql, (q['id'],))
        for row in rows:
            if row[1].startswith('TmTag'):
                tags = row[2].split(';')
                for tag in tags:
                    tok = tag.split(':')
                    if len(tok) == 3 and tok[1] == 'MESH':
                        if tok[2].startswith('D'):  # only consider descriptors
                            c_pred['mesh'].add(tok[2])
            elif row[1] == 'MetaMap':
                c_pred['mesh'] |= set(row[2].split(';'))
            elif row[1] == 'GO':
                c_pred['go'] |= set(row[2].split(';'))

        expected = []
        for x in c_exp.values():
            expected.extend(x)

        returned = []
        for x in c_pred.values():
            returned.extend(list(x))
        table.clear_rows()
        q_fl = '[{}]\n{}'.format(q['id'], q['body'])
        table.add_row([q_fl,
                       '\n'.join(c_exp['mesh']),
                       '\n'.join(c_exp['go']),
                       '\n'.join(c_exp['uniprot']),
                       '\n'.join(c_exp['do']),
                       '\n'.join(returned)])
        print(table)
        prec = utils.precision(expected, returned)
        precisions.append(prec)
        recall = utils.recall(expected, returned)
        recalls.append(recall)
        f_score = utils.f_measure(prec, recall)
        f_scores.append(f_score)
        avg_precisions.append(utils.avg_precision(expected, returned))
        logger.info('prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'
                    ''.format(prec, recall, f_score))
    logger.info('mean_prec: {:.4f}, mean_recall: {:.4f}, mean_f1: {:.4f}'
                ''.format(sum(precisions) / len(sample_data),
                          sum(recalls) / len(sample_data),
                          sum(f_scores) / len(sample_data)))
    logger.info('map: {:.4f}, gmap: {:.4f}'
                ''.format(sum(avg_precisions) / len(sample_data),
                          math.exp(sum([math.log(x + 1e-6) for x in

                                        avg_precisions]) / len(sample_data))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        help="Path to the BioASQ training dataset (in JSON)")
    parser.add_argument('-s', '--sample_size', type=float,
                        help='sample of BioAsq training dataset;'
                             '< 1 percentage, >= 1 num. of sample, 0 all')
    parser.add_argument('-q', '--qids', nargs='*',
                        help="evaluate on this question only")
    parser.add_argument('--database', type=str,
                        help='path to the concepts database')
    args = parser.parse_args()

    # set defaults
    if not args.dataset:
        args.dataset = os.path.join(
            DATA_DIR, 'bioask/BioASQ-training5b/BioASQ-trainingDataset5b.json')
    if not args.sample_size:
        args.sample_size = .2  # 20% of 1,799 examples
    if not args.database:
        args.database = os.path.join(DATA_DIR, 'concepts.db')

    evaluate(args)