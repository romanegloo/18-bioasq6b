#!/usr/bin/env python3
"""A script to run QA_proximity model interactively."""

import os
import code
import argparse
import logging

from BioAsq6B import DATA_DIR
from BioAsq6B.qa_proximity import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--qaprox-model', type=str,
                    help='Path to qa_prox model to use')
args = parser.parse_args()

if args.model is None:
    args.model = os.path.join(DATA_DIR, 'qa_prox/var/best.mdl')

p = Predictor(args)

banner = """
QA_Proximity Model
- Run predict with a question and context
-------------------------------------------------------------------------
>> question = "Is the protein Papilin secreted?"
>> context = "Receptor activator of nuclear factor κB ligand (RANKL) is a
cytokine predominantly secreted by osteoblasts."
>> p.predict(question, context)
>> usage()
"""

def usage():
    print(banner)

code.interact(banner=banner, local=locals())