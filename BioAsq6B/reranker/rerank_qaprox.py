#!/usr/bin/env python3
"""use pre-trained QA_Proximity model to re-rank retrieved list of documents"""

import subprocess
import os
import spacy
import logging
from collections import OrderedDict
import numpy as np
from threading import Thread

from .. import DATA_DIR
from .. import common
from ..qa_proximity import Predictor

logger = logging.getLogger(__name__)

class RerankQaProx(object):
    def __init__(self, args, cached_scores=None):
        self.args = args
        self.idx_dir = os.path.join(DATA_DIR, 'galago-medline-full-idx')
        logger.info('loading spacy nlp tools...')
        self.nlp = spacy.load('en')
        self.predictor = Predictor(args)
        self.cached_scores = cached_scores

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
        if self.args.verbose:
            logger.info('analyzing qa relevance of qid #{}'.format(q['id']))
        rel_scores = []
        self.predictor._set_q(q['body'])
        for docid in docids:
            if self.args.cache_scores:
                # check if cached score exists
                key = q['id'] + '-' + docid
                if key in self.cached_scores:
                    rel_score = self.cached_scores[key]
                    rel_scores.append(rel_score)
                    continue

            # Proximity score will be the sum of the average prox scores and
            # the the best median of 3 consecutive scores.
            scores = []
            threads = []
            for sent, tokens in self.get_sentence(docid):
                t = Thread(target=self.predictor.predict_prob, args=(sent, ),
                           kwargs={'tokens': tokens, 'scores': scores})
                threads.append(t)
                t.start()
            for i, t in enumerate(threads):
                t.join()

            if len(scores) < 2:  # penalize an empty (or short) document
                rel_score = (0.1, 0.1, 0.1)
            else:
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                best_median = 0
                for i in range(len(scores)-2):
                    median = np.median(scores[i:i+3])
                    if best_median < median:
                        best_median = median
                if best_median == 0:
                    best_median = avg_score
                rel_score = (avg_score, best_median, max_score)
            rel_scores.append(rel_score)
            if self.args.cache_scores:
                key = q['id'] + '-' + docid
                self.cached_scores[key] = rel_score
            if self.args.verbose:
                print("reranking {} analyzed".format(docid))
        return rel_scores

    def get_sentence(self, docid):
        """read document and split by sentence, and yield each sentence"""
        text = self.read_doc_text(docid)
        if text is None:
            return
        s_ = self.nlp(text)
        for sent in s_.sents:
            yield sent.text, s_[sent.start:sent.end]

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
        # prepend the title; between <TITLE> and </TITLE>
        start = doc.find('<TITLE>') + len('<TITLE>')
        end = doc.find('</TITLE>')
        if start >= end or start <= len('<TITLE>') or end <= 0:
            return
        text = doc[start:end] + ' ' + text
        return text

    @staticmethod
    def merge_scores(weights, docids, ret_scores, rel_scores):
        score_weights = list(map(float, weights.split(',')))
        def _softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        ranked_list = {}
        # Normalize retrieval scores
        ret_scores_norm = _softmax(ret_scores) * len(ret_scores)
        for i, d in enumerate(docids):
            ranked_list[d] = {'ret_score': ret_scores[i]}
            try:
                ranked_list[d]['avg_rel_scores'] = \
                    np.average(rel_scores[i], weights=score_weights[:3])
            except TypeError:
                print(rel_scores[i])
                raise
            ranked_list[d]['rel_scores'] = rel_scores[i]
            ranked_list[d]['ret_score_norm'] = ret_scores_norm[i]
            ranked_list[d]['score'] = \
                score_weights[3] * ranked_list[d]['ret_score_norm'] + \
                score_weights[4] * ranked_list[d]['avg_rel_scores']
        # Sort
        results = OrderedDict(sorted(ranked_list.items(),
                                     key=lambda t: t[1]['score'], reverse=True))
        return results

