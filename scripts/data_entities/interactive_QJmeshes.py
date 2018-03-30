"""Given a query mesh term (CUI), get the list of journal meshes"""

from pathlib import Path
import pickle
from random import sample
import code
import logging
import sqlite3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

data_dir = Path(__file__).absolute().parents[3] / 'data'
table_file = data_dir / 'journal_mesh_dist.pkl'
desc_db = data_dir / 'concepts.db'

logger.info("Loading distribution table...")
data = pickle.load(table_file.open('rb'))
""" data keys:
['journals', 'qmesh2idx', 'dist_table', 'idx2qmesh', 'jmesh2idx', 'idx2jmesh']
"""

if data is None:
    raise RuntimeError("Failed loading")
table = data['dist_table']

def ex(k=5):
    qmesh_indices = sample(range(table.shape[0]), k)
    for qidx in qmesh_indices:
        qmesh = data['idx2qmesh'][qidx]
        print("{} \"{}\"".format(qmesh, get_mesh_name(qmesh)))
        for jmesh in [(i, cnt) for i, cnt in enumerate(table[qidx]) if cnt > 0]:
            jmesh_id = data['idx2jmesh'][jmesh[0]]
            print("  - {} \"{}\" ({})"
                  "".format(jmesh_id, get_mesh_name(jmesh_id), jmesh[1]))


def get_mesh_name(id):
    cnx = sqlite3.connect(desc_db.as_posix())
    csr = cnx.cursor()
    csr.execute("SELECT * FROM mesh WHERE cui='{}';".format(id))
    concept = csr.fetchone()
    if concept is not None:
        return concept[2]
    return 'N/A'


def get_jmeshes(cui):
    if cui in data['qmesh2idx']:
        qidx = data['qmesh2idx'][cui]
        print("{} \"{}\"".format(cui, get_mesh_name(cui)))
        for jmesh in [(i, cnt) for i, cnt in enumerate(table[qidx]) if cnt > 0]:
            jmesh_id = data['idx2jmesh'][jmesh[0]]
            print("  - {} \"{}\" ({})"
                  "".format(jmesh_id, get_mesh_name(jmesh_id), jmesh[1]))
    else:
        print('{} not found'.format(cui))
        return ''
banner = """
usage:
  >>> ex(k=5)  # returns random meshes with its mapping journal meshes
  >>> get_jmeshes('D006627')  # get the journal meshes by the given mesh cui
"""

def usage():
    print(banner)

code.interact(banner, local=locals())