#!/usr/bin/env python3

import os
from .. import PATHS
from ..tokenizers import SpacyTokenizer

DEFAULTS = {
    'tokenizer': SpacyTokenizer
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    """used to have [tfidf, sqlite, bioasq, solr, galago].
    Now I only use galago. The option list may extend later."""
    if name == 'galago':
        return GalagoRanker
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_ranker_galago import GalagoRanker
from .query_builder import QueryBuilder
