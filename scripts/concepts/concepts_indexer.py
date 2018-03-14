"""Via serveral concept related data services, store the retrieved concepts
by the BioAsq question id in a local sqlite3 database"""
import argparse
import os
import logging
import json
from pathlib import PosixPath
from BioAsq6B import PATHS
from BioAsq6B.data_services import ConceptRetriever

DATA_DIR = \
  os.path.join(PosixPath(__file__).absolute().parents[4].as_posix(), 'data')
parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str,
                    default=os.path.join(PATHS['train_dir'],
                                         'BioASQ-trainingDataset6b.json'),
                    help='Path to a BioASQ training file')
parser.add_argument('--database', type=str, default=PATHS['concepts_db'],
                    help='Path to the sqlite database file')
args = parser.parse_args()

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# Read training dataset
with open(args.input_file) as f:
    data = json.load(f)
    questions = data['questions']
logger.info('{} quetions read'.format(len(questions)))

# questions = questions[:3]
# Retrieve concepts
cr = ConceptRetriever(args, updateDatabase=True)
concepts = cr.get_concepts(questions)
