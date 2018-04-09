import os
import sys
from pathlib import PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('BioAsq6B supports Python 3.5 or higher.')

PATHS = {}
root_dir = PosixPath(__file__).absolute().parents[2].as_posix()
PATHS['data_dir'] = os.path.join(root_dir, 'data')
PATHS['runs_dir'] = os.path.join(root_dir, 'runs')
PATHS['test_dir'] = os.path.join(PATHS['data_dir'], 'bioasq/test')
PATHS['train_dir'] = os.path.join(PATHS['data_dir'], 'bioasq/train')
PATHS['galago_idx'] = os.path.join(PATHS['data_dir'], 'galago-medline-full-idx')
PATHS['concepts_db'] = os.path.join(PATHS['data_dir'], 'concepts.db')
PATHS['qasim_model'] = os.path.join(PATHS['data_dir'], 'qa_prox/var/best.mdl')
PATHS['idf_file'] = os.path.join(PATHS['data_dir'], 'qa_prox/idf.p')
PATHS['embedding_file'] = os.path.join(PATHS['data_dir'],
                          'qa_prox/embeddings/wikipedia-pubmed-and-PMC-w2v.bin')
# Cached Qa_Sim scores file
PATHS['cached_scores_file'] = \
    os.path.join(PATHS['runs_dir'], 'cached_scores.pkl')
PATHS['cached_docs_file'] = os.path.join(PATHS['runs_dir'], 'cached_docs.pkl')

from . import retriever
from .cache import Cache
from .data_services import BioasqAPI
