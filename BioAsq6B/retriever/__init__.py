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


from .doc_ranker_galago import GalagoSearch
from .query_builder import QueryBuilder
