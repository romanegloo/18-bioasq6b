#!/usr/bin/env python3
"""Query builder for various uses"""
from . import DEFAULTS
from . import utils
from .. import tokenizers
from ..data_services import MetamapExt


class QueryBuilder(object):
    """given query string (e.g. question body), build an expanded query
    string with respect to the retrieval types (e.g. solr or bioasq data
    services)"""

    def __init__(self, q_type='solr', tokenizer=None):
        self.q_type = q_type
        if not tokenizer:
            tokenizer_class = DEFAULTS['tokenizer']
        else:
            tokenizer_class = tokenizers.get_class(tokenizer)
        self.tokenizer = tokenizer_class()

    def _ngram_tokenize(self, query, ngram=2):
        tokens = self.tokenizer.tokenize(query.lower())
        ngrams = tokens.ngrams(n=ngram, uncased=True,
                               filter_fn=utils.filter_ngram)
        return ngrams

    def build(self, query):
        if self.q_type == 'api_docs':
            ngrams = self._ngram_tokenize(query)
            query = ' OR '.join(['"{}"'.format(t) for t in ngrams])
            return query
        elif self.q_type == 'solr':
            return self.solr_build_q(query)
        elif self.q_type == 'galago':
            return self.galago_build_q(query)

    def build_batch(self, queries):
        if self.q_type == 'galago':
            return self.galago_build_queries(queries)
        else:
            raise NotImplementedError

    def solr_build_q(self, query):
        ngrams = self._ngram_tokenize(query)
        query = ' '.join(['"{}"'.format(t) for t in ngrams])
        # search over meshHeading
        mm = MetamapExt()
        meshHeadings = mm.get_mesh_names(query)
        q_mesh = ' '.join(['meshHeading:"{}"'.format(mh)
                           for mh in meshHeadings])
        query += ' ' + q_mesh
        # filter out non-Pubmed articles
        query += ' -id:(AACR* OR ASCO*)'
        return query

