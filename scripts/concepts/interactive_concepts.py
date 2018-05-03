#!/usr/bin/python3

"""This script can be used to validate/build/update concepts database via
conceptRetriever"""

import code
import logging
import os
import json

from BioAsq6B.data_services import ConceptRetriever
from BioAsq6B import PATHS

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

retriever = ConceptRetriever()
train_file = os.path.join(PATHS['test_dir'], 'phaseA_6b_05.json')
questions = None
banner = """
usage:
  >>> update_metamap_results()
  ...
  >>> read_concepts_from_db(qid='55031181e9bde69634000014')
  ...
"""


def update_metamap_results():
    global questions
    if questions is None:
        logger.info('Reading training dataset...')
        with open(train_file) as f:
            data = json.load(f)
            questions = data['questions']
    retriever.run_metamap(questions)
    retriever.update_database()


def test(sec=0):
    global questions
    with open(train_file) as f:
        data = json.load(f)
        questions = data['questions'][25*sec:25*(sec+1)]
    retriever.updateDatabase = True
    # retriever.read_from_db(questions)
    # retriever.run_go_expand()
    # retriever.update_database()
    retriever.get_concepts(questions)


def read_concepts_from_db(qid):
    concepts = retriever.get_concepts([{'id': qid}])
    print(concepts)


def usage():
    print(banner)


code.interact(banner, local=locals())
