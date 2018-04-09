"""Testing Cache object"""

import code
import logging
import pickle
import os
import regex
import numpy as np

from BioAsq6B import Cache, PATHS

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

cache = Cache()

def t1():
    """Script to transform previously cached data (scores)"""
    keys_ = ['rankings', 'scores-ret', 'scores-qasim', 'scores-journal']
    # Read previous cache file
    logger.info("Reading cache scores file...")
    file = os.path.join(PATHS['runs_dir'], 'cached_scores.pkl')
    data = pickle.load(open(file, 'rb'))
    for key, entry in data.items():
        if key.endswith('-ret'):
            qid = key.split('-')[0]
            if qid not in cache.scores:
                cache.scores[qid] = dict.fromkeys(keys_)
            cache.scores[qid]['rankings'] = entry[0]
            cache.scores[qid]['scores-ret'] = entry[1]
            cache.scores[qid]['scores-qasim'] = {}
        elif regex.match(r'^.+-\d+$', key):
            qid, docid = key.split('-')
            if qid not in cache.scores:
                cache.scores[qid] = dict.fromkeys(keys_)
            if type(entry) == np.ndarray:
                entry = entry.tolist()
            # if docid not in cache.scores[qid]['scores-qasim']:
            try:
                cache.scores[qid]['scores-qasim'][docid] = entry
            except TypeError:
                cache.scores[qid]['scores-qasim'] = {'docid': entry}
    cache.save_scores()

def t2():
    """Script to transform previously cached data (docs)"""
    keys_ = ['title', 'text', 'journal']
    # Read from file
    logger.info("Reading cached docs file...")
    file = os.path.join(PATHS['runs_dir'], 'cached_docs.pkl')
    data = pickle.load(open(file, 'rb'))
    for key, entry in data.items():
        cache.documents[key] = entry
    cache.save_docs()


banner = """
usage:
  >>> cache.get_scores_retrieval('5a6d1db1b750ff4455000033', k=15)
"""
def usage():
    print(banner)

code.interact(banner, local=locals())
