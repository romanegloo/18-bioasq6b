import os
import sys
from pathlib import PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('BioAsq6B supports Python 3.5 or higher.')

PATHS = {}
root_dir = PosixPath(__file__).absolute().parents[2].as_posix()
PATHS['root_dir'] = root_dir
PATHS['data_dir'] = os.path.join(root_dir, 'data')
PATHS['runs_dir'] = os.path.join(root_dir, 'runs')
PATHS['test_dir'] = os.path.join(PATHS['data_dir'], 'bioasq/test')
PATHS['train_dir'] = os.path.join(PATHS['data_dir'], 'bioasq/train')
PATHS['galago_idx'] = os.path.join(PATHS['data_dir'], 'galago-medline-full-idx')
PATHS['concepts_db'] = os.path.join(PATHS['data_dir'], 'concepts.db')
PATHS['qasim_model'] = os.path.join(PATHS['data_dir'],
                                    'qa_prox/var/best-nofeatures.mdl')
PATHS['idf_file'] = os.path.join(PATHS['data_dir'], 'qa_prox/idf.p')
# PATHS['embedding_file'] = \
#     '/scratch/jno236-data/word_embeddings/fastText_pretrained/wiki.en.vec'
PATHS['embedding_file'] = '/scratch/jno236-data/word_embeddings/bionlp/' \
                          'wikipedia-pubmed-and-PMC-w2v.bin'
PATHS['mysqldb_cred_file'] = os.path.join(PATHS['data_dir'], 'db_cred.csv')
PATHS['metamap_bin'] = '/home/jno236/opt/public_mm/bin/metamap'
PATHS['ranklib_bin'] = os.path.join(root_dir, 'opt/RankLib-2.1-patched.jar')

from . import retriever
from .data_services import BioasqAPI
