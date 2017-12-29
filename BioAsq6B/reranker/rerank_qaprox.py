#!/usr/bin/env python3
"""use pre-trained QA_Proximity model to re-rank retrieved list of documents"""

import subprocess
import os
import spacy
import logging
from tqdm import tqdm
from collections import OrderedDict
import math

from .. import DATA_DIR
from ..qa_proximity import Predictor

logger = logging.getLogger(__name__)


# this case should reserve the order
# +------------------------------------------+----------+-----------------------------+
# | Question                                 |    GT    |                    Returned |
# +------------------------------------------+----------+-----------------------------+
# | [53482bfcaeec6fbd07000010]               | 24195105 | 24195105 [-15.8868, 0.2332] |
# | What are the characteristics of the      | 20823122 | 20823122 [-16.5059, 0.7841] |
# | "Universal Proteomics Standard 2"        |          | 25495225 [-16.7098, 0.4275] |
# | (UPS2)?                                  |          | 26545397 [-16.7116, 0.3183] |
# |                                          |          | 27354379 [-16.7674, 0.6461] |
# |                                          |          | 27241913 [-16.7950, 0.5309] |
# |                                          |          | 20811336 [-16.8261, 0.9980] |
# |                                          |          | 20657548 [-16.8292, 0.4598] |
# |                                          |          | 27354376 [-16.8336, 0.2314] |
# |                                          |          | 21930701 [-16.8963, 0.9735] |
# |                                          |          | 26457653 [-16.9000, 0.1046] |
# |                                          |          | 18370308 [-16.9169, 0.1741] |
# |                                          |          | 25883932 [-16.9173, 0.4659] |
# |                                          |          | 20622808 [-16.9237, 0.3848] |
# |                                          |          | 19506038 [-16.9419, 0.6576] |
# |                                          |          | 23584085 [-16.9470, 0.4306] |
# |                                          |          | 15308648 [-16.9543, 0.5936] |
# |                                          |          | 21887821 [-16.9544, 0.9973] |
# |                                          |          | 25143245 [-16.9567, 0.4426] |
# |                                          |          | 23124206 [-16.9587, 0.3922] |
# |                                          |          | 26649523 [-16.9655, 0.1322] |
# |                                          |          | 26037908 [-16.9675, 0.3472] |
# |                                          |          | 16219938 [-16.9730, 0.1940] |
# |                                          |          | 21751410 [-16.9779, 0.2271] |
# |                                          |          | 22403410 [-16.9828, 0.4954] |
# |                                          |          | 22522798 [-16.9910, 0.5007] |
# |                                          |          | 21250827 [-16.9925, 0.3320] |
# |                                          |          | 12294134 [-16.9944, 0.9823] |
# |                                          |          | 27039242 [-17.0045, 0.2626] |
# |                                          |          | 19674966 [-17.0146, 0.4951] |
# +------------------------------------------+----------+-----------------------------+


class RerankQaProx(object):
    def __init__(self, args):
        self.args = args
        if args.score_lambda:
            self.score_lambda = args.score_lambda
        else:
            self.score_lambda = 1.5
        self.idx_dir = os.path.join(DATA_DIR, 'galago-medline-full-idx')
        logger.info('loading spacy nlp tools...')
        self.nlp = spacy.load('en')
        self.predictor = Predictor(args)

    def batch_get_prox_scores(self, lst_docids, questions):
        """Get QA_Proximity scores in batch; return the list of rel_scores"""
        # For each document, run prox model on the sentences
        lst_rel_scores = []
        for q, docids in zip(questions, lst_docids):
            rel_scores = self.get_prox_scores(docids, q)
            lst_rel_scores.append(rel_scores)
        return lst_rel_scores

    def get_prox_scores(self, docids, q):
        """Compute the QA_Proximity scores over the list of documents"""
        logger.info('analyzing qa relevance of qid #{}\n'.format(q['id']))
        rel_scores = []
        for docid in tqdm(docids):
            # todo. might need to read best N sentences and scores,
            # or multiple sentences and combined scores
            best = {'score': 0., 'context': ''}
            for sent in self.get_sentence(docid):
                prob = self.predictor.predict_prob(q['body'], sent)
                if prob[1] > best['score']:
                    best['score'] = prob[1]
                    best['context'] = sent
            rel_scores.append(best['score'])
        return rel_scores

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

    def merge_scores(self, docids, ret_scores, rel_scores=None):
        ranked_list = {}
        # Normalize retrieval scores
        ret_scores_norm = self._softmax(ret_scores)
        for i, d in enumerate(docids):
            ranked_list[d] = { 'ret_score': ret_scores[i] }
            if rel_scores:
                ranked_list[d]['rel_score'] = rel_scores[i]
                ranked_list[d]['score'] = \
                    self.score_lambda * ret_scores_norm[i] * rel_scores[i]
            else:
                ranked_list[d]['score'] = \
                    self.score_lambda * ret_scores_norm[i]
        # Sort
        results = OrderedDict(sorted(ranked_list.items(),
                                     key=lambda t: t[1]['score'], reverse=True))
        return results

    def _softmax(self, x):
        import numpy as np
        return np.exp(x) / np.sum(np.exp(x), axis=0)
