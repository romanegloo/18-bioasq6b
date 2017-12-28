#!/usr/bin/env python3
"""use pre-trained QA_Proximity model to re-rank retrieved list of documents"""

import subprocess
import os
import spacy
import logging
from tqdm import tqdm

from .. import DATA_DIR
from ..qa_proximity import Predictor

logger = logging.getLogger(__name__)


class RerankQaProx(object):
    def __init__(self, args):
        self.args = args
        self.idx_dir = os.path.join(DATA_DIR, 'galago-medline-full-idx')
        logger.info('loading spacy nlp tools...')
        self.nlp = spacy.load('en')
        self.predictor = Predictor(args)

    def run(self, lst_docids, lst_scores, questions):
        """Re-rank the list of documents by the Qa_Proximity scores"""

        # For each document, run prox model on the sentences
        lst_rel_scores = []
        for q, lst_docid in zip(questions, lst_docids):
            q_body = q['body']
            rel_scores = []
            logger.info('analyzing qa relevance of qid #{}'.format(q['id']))
            for docid in tqdm(lst_docid):
                # todo. might need to read best N sentences and scores,
                # or multiple sentneces and combined scores
                best = {'score': 0., 'context': ''}
                for sent in self.get_sentence(docid):
                    prob = self.predictor.predict_prob(q_body, sent)
                    if prob[1] > best['score']:
                        best['score'] = prob[1]
                        best['context'] = sent
                rel_scores.append(best['score'])
            lst_rel_scores.append(rel_scores)
        return lst_rel_scores

    def get_sentence(self, docid):
        """read document and split by sentence, and yield each sentence"""
        text = self.read_doc_text(docid)
        if text is None:
            return
        s_ = self.nlp(text)
        for sent in s_.sents:
            yield sent.text

    def read_doc_text(self, docid):
        p = subprocess.run(['galago', 'doc', '--index={}'.format(self.idx_dir),
                          '--id=PMID-{}'.format(docid)], stdout=subprocess.PIPE)
        doc = p.stdout.decode('utf-8')
        # find the abstract; between <TEXT> and </TEXT>
        start = doc.find('<TEXT>') + len('<TEXT>')
        end = doc.find('</TEXT>')
        if start >= end or start <= len('<TEXT>') or end <= 0:
            return
        text = doc[start:end]
        return text

