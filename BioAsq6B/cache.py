"""Interface for using cached data (scores, documents, concepts, etc.)"""
import pickle
import os
import logging

from BioAsq6B import PATHS

logger = logging.getLogger()


class Cache(object):
    docs_file = os.path.join(PATHS['data_dir'], 'cache-docs.pkl')
    scores_file = os.path.join(PATHS['data_dir'], 'cache-scores.pkl')

    def __init__(self, args=None):
        self.args = args
        self.documents = dict()
        self.scores = dict()
        self.score_keys = ['retrieval']  # qasim/journal can be appended
        self.flg_update_scores = {'retrieval': False}
        self.scores_changed = False
        self.documents_changed = False

        # Set update flags
        self.confirm_updates()
        # Read-in
        self.read_from_cache_files()

    def confirm_updates(self):
        if self.args is None:
            return
        for reranker in self.args.rerank:
            self.score_keys.append(reranker)
            self.flg_update_scores[reranker] = False
        if self.args.verbose:
            for score_scheme in self.score_keys:
                ans = input("Update {} scores [Y/n]? ".format(score_scheme))
                if not ans.lower().startswith('n'):
                    self.flg_update_scores[score_scheme] = True

    def read_from_cache_files(self):
        # Read cache-docs
        if os.path.exists(Cache.docs_file):
            logger.info("Reading cached-docs...")
            self.documents = pickle.load(open(self.docs_file, 'rb'))
        # Read cache-scores
        if os.path.exists(Cache.scores_file):
            logger.info("Reading cached-scores...")
            self.scores = pickle.load(open(self.scores_file, 'rb'))

    def get_scores_retrieval(self, qid, k=None):
        if qid in self.scores:
            if k is not None:
                return self.scores[qid]['rankings'][:k], \
                       self.scores[qid]['scores-ret'][:k]
            else:
                return self.scores[qid]['rankings'], \
                       self.scores[qid]['scores-ret']
        return [], []

    def update_scores(self, res):
        """get a res (RankedDocs), update scores if needed"""
        qid = res.query['id']
        if qid not in self.scores:
            self.scores[qid] = {}
        if 'retrieval' in res.update_cache:
            self.scores[qid]['rankings'] = res.rankings
            self.scores[qid]['expected-docs'] = res.expected_docs
            self.scores[qid]['scores-ret'] = res.scores['retrieval']
            self.scores_changed = True
        if 'qasim' in res.update_cache:
            if 'scores-qasim' not in self.scores[qid] or \
                    type(self.scores[qid]['scores-qasim']) != dict:
                self.scores[qid]['scores-qasim'] = dict()
            for i, docid in enumerate(res.rankings):
                self.scores[qid]['scores-qasim'][docid] =\
                    res.scores['qasim'][i]
            self.scores_changed = True
        if 'journal' in res.update_cache:
            self.scores[qid]['scores-journal'] = res.scores['journal']
            self.scores_changed = True
        if 'semmeddb' in res.update_cache:
            self.scores[qid]['scores-semmeddb'] = res.scores['semmeddb']
            self.scores_changed = True
        if 'documents' in res.update_cache:
            for docid in res.rankings:
                self.documents[docid] = res.docs_data[docid]
            self.documents_changed = True

    def save_scores(self):
        logger.info('Saving {} entries of cached scores...'
                    ''.format(len(self.scores)))
        pickle.dump(self.scores, open(self.scores_file, 'wb'))

    def save_docs(self):
        logger.info('Saving {} entries of cached docs...'
                    ''.format(len(self.documents)))
        pickle.dump(self.documents, open(self.docs_file, 'wb'))

