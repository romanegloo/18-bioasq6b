#!/usr/bin/python3

"""This script is to interact with SemMedDB which contains relations between
concepts observed in pubmed literature.

Features:
- By given set of UMLS concepts, return the predications of a subject and an
objects that match the concepts
- By given pubmed id (PMID) and a BioASQ question, find if the CUIs of the
questions exist in the SemMedDB entries. Return the score"""

import code
import json
import os
import logging

from BioAsq6B import PATHS
from BioAsq6B.reranker import RerankerSemMedDB

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

semmeddb = RerankerSemMedDB()

banner = """
usage:
  >>> semmeddb.get_predications(qid='5118dd1305c10fae75000001')

To get a set of relevant documents to the question concepts
  >>> semmeddb.get_candidate_docs(qid='5118dd1305c10fae75000001')
  or
  >>> semmeddb.get_candidate_docs(cuis=['C0022646', 'C0000768', 'C0006104'])
"""

def run_stats():
    # read training dataset
    train_file = os.path.join(PATHS['train_dir'],
                              'BioASQ-trainingDataset6b.json')
    counts = {
        'total_question': 0,
        'predication_occurrences': 0,
        'document_hits': 0,
        'average_hit_ratio': 0
    }
    with open(train_file) as f:
        questions = json.load(f)['questions']
        counts['total_question'] = len(questions)

    for i, entry in enumerate(questions):
        print("question id: {}".format(entry['id']))
        res = semmeddb.get_candidate_docs(entry['id'])
        if len(res) > 0:
            counts['predication_occurrences'] += 1
            gt_docs = [doc_url.split('/')[-1] for doc_url in entry['documents']]
            ret_docs = [item[0] for item in res]
            ratio = len(set(gt_docs) & set(ret_docs)) / len(gt_docs)
            if ratio > 0:
                counts['document_hits'] += 1
                counts['average_hit_ratio'] += ratio
            print("hit ratio: {}".format(ratio))
    counts['average_hit_ratio'] /= counts['document_hits']
    print(counts)


def run(qid):
    semmeddb.get_predications(qid)

def usage():
    print(banner)


code.interact(banner, local=locals())