#!/usr/bin/env python3
"""use pre-trained QA_Sim model to re-rank retrieved list of documents"""

import logging
import pickle

from BioAsq6B import PATHS
from BioAsq6B.qa_proximity import Predictor

logger = logging.getLogger()


class RerankQaSim(object):
    def __init__(self, args, nlp=None):
        self.args = args
        self.nlp = nlp
        self.predictor = Predictor(args, nlp=nlp)

    def get_qasim_scores(self, ranked_docs, cache=None):
        """Compute the QA_Sim scores over the list of documents.
        Also while computing QA_Sim scores, get top 10 relevant
        sentences and its offsets"""
        q = ranked_docs.query
        self.predictor.set_q(q['body'], q['type'])
        # If cache is given, read cached scores and extract snippets
        k = len(ranked_docs.rankings)
        # Validate qasim_scores; Qasim score for a docid should be given in
        # an array format
        scores_ = []
        snippets_ = []
        for docid in ranked_docs.rankings:
            use_cache_scores = True
            if cache is None:
                use_cache_scores = False
            if not use_cache_scores or 'scores-qasim' not in cache \
                    or type(cache['scores-qasim']) != dict\
                    or docid not in cache['scores-qasim']:
                use_cache_scores = False
            if not use_cache_scores \
                    or type(cache['scores-qasim'][docid]) != list:
                use_cache_scores = False
            if use_cache_scores:
                s = cache['scores-qasim'][docid]
                scores_.append(s)
            else:
                logger.info('qasim for qid {} docid {}'.format(q['id'], docid))
                if 'qasim' not in ranked_docs.update_cache:
                    ranked_docs.update_cache.append('qasim')
                s, t = self.predictor.predict_prob_b(
                    ranked_docs.docs_data[docid]['text'],
                    ranked_docs.docs_data[docid]['title'],
                    docid=docid
                )
                scores_.append(s)
                snippets_.extend(t)
        ranked_docs.scores['qasim'] = scores_[:k]
        ranked_docs.text_snippets = \
            sorted(snippets_, key=lambda t: t[1], reverse=True)

    def get_sentence(self, docid, text):
        """read document and split by sentence, and yield each sentence"""
        if text is None:
            return
        s_ = self.nlp(text)
        for sent in s_.sents:
            yield sent.text, s_[sent.start:sent.end]

    # def merge_scores(self, weights, docids, ret_scores, rel_scores):
    #     score_weights = list(map(float, weights.split(',')))
    #     ranked_list = {}  # Container to store different scores by docid
    #     assert len(docids) == len(rel_scores)
    #     if self.args.score_fusion == 'weighted_sum' and \
    #             type(rel_scores[0]) == tuple:
    #         assert len(score_weights) == 4
    #         # Normalize retrieval scores
    #         ret_scores_norm = len(ret_scores) * \
    #             np.exp(ret_scores) / np.sum(np.exp(ret_scores), axis=0)
    #         for i, d in enumerate(docids):
    #             ranked_list[d] = {'ret_score': ret_scores[i]}
    #             ranked_list[d]['avg_rel_scores'] = \
    #                 np.average(rel_scores[i], weights=score_weights[:3])
    #             ranked_list[d]['rel_scores'] = rel_scores[i]
    #             ranked_list[d]['ret_score_norm'] = ret_scores_norm[i]
    #             ranked_list[d]['score'] = \
    #                 np.average([ret_scores_norm[i]] + list(rel_scores[i]),
    #                            weights=score_weights)
    #         # Sort
    #         results = OrderedDict(sorted(ranked_list.items(),
    #                                      key=lambda t: t[1]['score'],
    #                                      reverse=True))
    #         return results
    #
    #     if self.args.score_fusion == 'weighted_sum' and \
    #             type(rel_scores[0]) != tuple:
    #         # Normalize retrieval scores
    #         ret_scores_norm = len(ret_scores) * \
    #                      np.exp(ret_scores) / np.sum(np.exp(ret_scores), axis=0)
    #         for i, d in enumerate(docids):
    #             ranked_list[d] = {'ret_score': ret_scores[i]}
    #             ranked_list[d]['avg_rel_scores'] = rel_scores[i]
    #             ranked_list[d]['rel_scores'] = rel_scores[i]
    #             ranked_list[d]['ret_score_norm'] = ret_scores_norm[i]
    #             ranked_list[d]['score'] = \
    #                 score_weights[0] * ret_scores_norm[i] +\
    #                 score_weights[1] * rel_scores[i]
    #         # Sort
    #         results = OrderedDict(sorted(ranked_list.items(),
    #                                      key=lambda t: t[1]['score'],
    #                                      reverse=True))
    #         return results
    #
    #     if self.args.score_fusion == 'rrf':
    #         """Reciprocal Ranks Fusion (with parameterized weights)"""
    #         assert len(score_weights) == 4
    #         ret_r = [r for r in range(len(ret_scores))]
    #         rel1_r = sorted(range(len(rel_scores)),
    #                         key=lambda k: rel_scores[k][0], reverse=True)
    #         rel2_r = sorted(range(len(rel_scores)),
    #                         key=lambda k: rel_scores[k][1], reverse=True)
    #         rel3_r = sorted(range(len(rel_scores)),
    #                         key=lambda k: rel_scores[k][2], reverse=True)
    #
    #         for i, v in enumerate(zip(docids, ret_r, rel1_r, rel2_r, rel3_r)):
    #             ranked_list[v[0]] = {'ret_score': ret_scores[i]}
    #             ranked_list[v[0]]['rel_scores'] = rel_scores[i]
    #             ranked_list[v[0]]['score'] = \
    #                 sum(map(
    #                     lambda ix_s: 1 / (score_weights[ix_s[0]] + ix_s[1]),
    #                     enumerate(v[1:])))
    #         results = OrderedDict(sorted(ranked_list.items(),
    #                                      key=lambda t: t[1]['score'],
    #                                      reverse=True))
    #         return results
    #     print('no results')

    def save_scores(self):
        logger.info('saving cached scores file')
        with open(PATHS['cached_scores_file'], 'wb') as f:
            pickle.dump(dict(self.cached_scores), f)

