"""Given a query mesh term (CUI), get the list of journal meshes"""

from random import sample
import code
import logging
from collections import namedtuple

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


def usage():
    print(banner)

code.interact(banner, local=locals())