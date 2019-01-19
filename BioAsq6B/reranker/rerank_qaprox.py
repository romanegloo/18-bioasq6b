#!/usr/bin/env python3
"""use pre-trained QA_Sim model to re-rank retrieved list of documents"""

import logging

from BioAsq6B.QaSimSent import Predictor

logger = logging.getLogger()


class RerankQaSim(object):
    def __init__(self, args, nlp=None):
        self.args = args
        self.nlp = nlp
        self.predictor = Predictor(args, nlp=nlp)
        if args.mode == 'test':
            self.predictor.model.load_embeddings()

    def get_qasim_scores(self, ranked_docs):
        """Compute the QA_Sim scores over the list of documents.
        Also while computing QA_Sim scores, get top 10 relevant
        sentences and its offsets"""
        q = ranked_docs.query
        self.predictor.set_q(q['body'], q['type'])
        # k = len(ranked_docs.rankings)
        # Validate qasim_scores; Qasim score for a docid should be given in
        # an array format
        results = []
        for docid in ranked_docs.rankings:
            print('qasim evaluating docid {}\r'.format(docid), end='')
            # Predict on the body of the document (ignore title)
            result = self.predictor.predict_prob_b(
                ranked_docs.docs_data[docid]['text'],
                docid=docid
            )
            results.append(result)
        scores_ = [[r['score'] for r in d] for d in results]
        ranked_docs.scores['qasim'] = scores_
        # sort_idx = sorted(range(len(scores_)), key=lambda k: scores_[k],
        #                   reverse=True)
        # ranked_docs.text_snippets = [results[i][0]['text'] for i in sort_idx]

    def get_sentence(self, docid, text):
        """read document and split by sentence, and yield each sentence"""
        if text is None:
            return
        s_ = self.nlp(text)
        for sent in s_.sents:
            yield sent.text, s_[sent.start:sent.end]

