#!/usr/bin/env python3
"""use pre-trained QA_Proximity model to re-rank retrieved list of documents"""

import subprocess
import spacy
import logging
import traceback
from collections import OrderedDict
import numpy as np
import pickle
from threading import Thread

from .. import PATHS
from ..qa_proximity import Predictor

logger = logging.getLogger()


class RerankQaProx(object):
    def __init__(self, args, cached_scores=None):
        self.args = args
        logger.info('loading spacy nlp tools...')
        self.nlp = spacy.load('en')
        self.predictor = Predictor(args)
        self.cached_scores = cached_scores

    def get_prox_scores(self, docids, q):
        """Compute the QA_Proximity scores over the list of documents.
        Also while computing QA_Proximity scores, get top 10 relevant
        sentences and its offsets"""
        assert len(docids) > 0
        if self.args.verbose:
            logger.info('analyzing qa relevance of qid #{}'
                        ''.format(q['id']))
        rel_scores = []
        agg_snippets = []
        try:
            self.predictor._set_q(q['body'], q['type'])
        except Exception as e:
            logger.error(traceback.format_exc())
            raise
        for docid in docids:
            if self.args.use_cache_scores:
                # check if cached score exists
                key = q['id'] + '-' + docid
                if key in self.cached_scores:
                    rel_score = self.cached_scores[key]
                    rel_scores.append(rel_score)
                    continue
            # Proximity score will be the sum of the average prox scores and
            # the the best median of 3 consecutive scores.
            doc_text, doc_title = self.read_doc_text(docid)
            scores, snippets = \
                self.predictor.predict_prob_b(doc_text, doc_title, docid=docid)
            agg_snippets.extend(snippets)
            agg_snippets = sorted(agg_snippets, key=lambda x: x[1],
                                  reverse=True)
            rel_scores.append(np.max(scores))
            if self.args.use_cache_scores:
                key = q['id'] + '-' + docid
                self.cached_scores[key] = np.max(scores)
            if self.args.verbose:
                logger.info("document {} done".format(docid))
        return rel_scores, agg_snippets

    def get_sentence(self, docid, text):
        """read document and split by sentence, and yield each sentence"""
        # text = self.read_doc_text(docid)
        if text is None:
            return
        s_ = self.nlp(text)
        for sent in s_.sents:
            yield sent.text, s_[sent.start:sent.end]

    def read_doc_text(self, docid):
        p = subprocess.run(['galago', 'doc',
                            '--index={}'.format(PATHS['galago_idx']),
                          '--id=PMID-{}'.format(docid)], stdout=subprocess.PIPE)
        doc = p.stdout.decode('utf-8')
        # find the abstract; between <TEXT> and </TEXT>
        start = doc.find('<TEXT>') + len('<TEXT>')
        end = doc.find('</TEXT>')
        if start >= end or start <= len('<TEXT>') or end <= 0:
            return
        text = doc[start:end]
        start = doc.find('<TITLE>') + len('<TITLE>')
        end = doc.find('</TITLE>')
        if start >= end or start <= len('<TITLE>') or end <= 0:
            return
        title = doc[start:end]
        return text, title

    def merge_scores(self, weights, docids, ret_scores, rel_scores):
        score_weights = list(map(float, weights.split(',')))
        ranked_list = {}  # Container to store different scores by docid
        assert len(docids) == len(rel_scores)
        if self.args.score_fusion == 'weighted_sum' and \
                type(rel_scores[0]) == tuple:
            assert len(score_weights) == 4
            # Normalize retrieval scores
            ret_scores_norm = len(ret_scores) * \
                np.exp(ret_scores) / np.sum(np.exp(ret_scores), axis=0)
            for i, d in enumerate(docids):
                ranked_list[d] = {'ret_score': ret_scores[i]}
                ranked_list[d]['avg_rel_scores'] = \
                    np.average(rel_scores[i], weights=score_weights[:3])
                ranked_list[d]['rel_scores'] = rel_scores[i]
                ranked_list[d]['ret_score_norm'] = ret_scores_norm[i]
                ranked_list[d]['score'] = \
                    np.average([ret_scores_norm[i]] + list(rel_scores[i]),
                               weights=score_weights)
            # Sort
            results = OrderedDict(sorted(ranked_list.items(),
                                         key=lambda t: t[1]['score'],
                                         reverse=True))
            return results

        if self.args.score_fusion == 'weighted_sum' and \
                type(rel_scores[0]) != tuple:
            # Normalize retrieval scores
            ret_scores_norm = len(ret_scores) * \
                         np.exp(ret_scores) / np.sum(np.exp(ret_scores), axis=0)
            for i, d in enumerate(docids):
                ranked_list[d] = {'ret_score': ret_scores[i]}
                ranked_list[d]['avg_rel_scores'] = rel_scores[i]
                ranked_list[d]['rel_scores'] = rel_scores[i]
                ranked_list[d]['ret_score_norm'] = ret_scores_norm[i]
                ranked_list[d]['score'] = \
                    score_weights[0] * ret_scores_norm[i] +\
                    score_weights[1] * rel_scores[i]
            # Sort
            results = OrderedDict(sorted(ranked_list.items(),
                                         key=lambda t: t[1]['score'],
                                         reverse=True))
            return results

        if self.args.score_fusion == 'rrf':
            """Reciprocal Ranks Fusion (with parameterized weights)"""
            assert len(score_weights) == 4
            ret_r = [r for r in range(len(ret_scores))]
            rel1_r = sorted(range(len(rel_scores)),
                            key=lambda k: rel_scores[k][0], reverse=True)
            rel2_r = sorted(range(len(rel_scores)),
                            key=lambda k: rel_scores[k][1], reverse=True)
            rel3_r = sorted(range(len(rel_scores)),
                            key=lambda k: rel_scores[k][2], reverse=True)

            for i, v in enumerate(zip(docids, ret_r, rel1_r, rel2_r, rel3_r)):
                ranked_list[v[0]] = {'ret_score': ret_scores[i]}
                ranked_list[v[0]]['rel_scores'] = rel_scores[i]
                ranked_list[v[0]]['score'] = \
                    sum(map(
                        lambda ix_s: 1 / (score_weights[ix_s[0]] + ix_s[1]),
                        enumerate(v[1:])))
            results = OrderedDict(sorted(ranked_list.items(),
                                         key=lambda t: t[1]['score'],
                                         reverse=True))
            return results
        print('no results')

