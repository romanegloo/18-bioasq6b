import os
import sys
from pathlib import PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('BioAsq6B supports Python 3.5 or higher.')

DATA_DIR = (
    os.getenv('BIOASQ_DATA') or
    os.path.join(PosixPath(__file__).absolute().parents[2].as_posix(), 'data')
)

from . import retriever
from .data_services import BioasqAPI
