"""Given a query mesh term (CUI), get the list of journal meshes"""

from pathlib import Path
import pickle
from random import sample
import code
import logging
import sqlite3

from BioAsq6B.reranker import RerankerJournal

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

journal_ranker = RerankerJournal()


def ex(k=5):
    qmesh_indices = sample(range(journal_ranker.dist.shape[0]), k)
    for qidx in qmesh_indices:
        qmesh = journal_ranker.data['idx2qmesh'][qidx]
        print("{} \"{}\"".format(qmesh, journal_ranker.get_mesh_name(qmesh)))
        for jmesh in [(i, cnt) for i, cnt in enumerate(journal_ranker.dist[qidx])
                      if cnt > 0]:
            jmesh_id = journal_ranker.data['idx2jmesh'][jmesh[0]]
            print("  - {} \"{}\" ({})"
                  "".format(jmesh_id,
                            journal_ranker.get_mesh_name(jmesh_id), jmesh[1]))

def get_jmeshes(qmesh):
    return journal_ranker.get_jmeshes(qmesh)
banner = """

usage:
  >>> ex(k=5)  # returns random meshes with its mapping journal meshes
  >>> get_jmeshes('D006627')  # get the journal meshes by the given mesh cui
"""

def f():
    ranked_docs = {
        'query': {
            'id': '589a245a78275d0c4a000026',
            'body': 'Elaborate on the link between conserved noncoding '
                    'elements (CNEs) and fractality.'
        },
        'rankings': [ '26899868', '24787386', '18045502', '26744417',
            '27412606', '21478460', '28874668', '24339797', '19492354',
            '19549339' ],
        'docs_data': {
            '26899868': {
                'journal': 'NIH Guide Grants Contracts'
            }
        }
    }
    journal_ranker.get_journal_scores(ranked_docs)

def usage():
    print(banner)

code.interact(banner, local=locals())