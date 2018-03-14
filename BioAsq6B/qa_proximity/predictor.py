#!/usr/bin/env python3
"""QA_Proximity predictor; Classifies if the given text is relevant to the
question."""

import logging
import spacy
import torch
import torch.nn.functional as F
from .model import QaProx
from .. import PATHS

logger = logging.getLogger()


class Predictor(object):
    """Load a pretrained model and predict inputs"""
    def __init__(self, args, idf=None):
        logger.info('Initializing model...')
        self.model = QaProx.load(PATHS['qasim_model'])
        # The model params may not be consistent with the user given params
        self.use_idf = self.model.args.use_idf
        self.no_token_feature = self.model.args.no_token_feature
        if self.no_token_feature:
            from spacy.tokenizer import Tokenizer
            nlp = spacy.load('en')
            self.nlp = Tokenizer(nlp.vocab)  # Use just Tokenizer
        else:
            self.nlp = spacy.load('en')  # Full nlp pipeline
        self.idf = idf

    def _set_q(self, q):
        self.q_ex, self.q_f, self.q_mask = self._encode_ex(q)

    def predict_prob(self, context, question=None, tokens=None, scores=None,
                     snippets=None):
        """
        Computes the probability of the context being close to the given
        question.
        :param context: a nlp parsed candidate answer sentence
        :param question: a nlp parsed question
        :param tokens: tokens with annotations
        :param scores: a score container
        :return:
        """
        if question:
            self._set_q(question)
        ex = self._build_ex(context, tokens)
        pred = self.model.predict(ex)
        res = F.sigmoid(pred).data.squeeze()
        # res = F.softmax(pred, 1).data.squeeze()
        if scores is not None:
            scores.append(res)
        if snippets is not None:
            entry = [{'document': '',
                      'text': context,
                      'offsetInBeginSection': 0,
                      'offsetInEndSection': 0,
                      'beginSection': 'abstract',
                      'endSection': 'abstract'}, res[0]]
            snippets.append(entry)
        return res

    def _encode_ex(self, sent, tokens=None):
        if self.no_token_feature:
            """No need to run nlp parser"""
            if tokens is None:
                tokens = self.nlp(sent)
            ex = dict()
            ex['context'] = [t.text.lower() for t in tokens]
            c_text = [self.model.word_dict[w] for w in ex['context']]
            x1 = torch.LongTensor(c_text).unsqueeze(0)
            x1_f = None
            x1_mask = torch.ByteTensor(1, len(ex['context'])).fill_(0)
            return x1, x1_f, x1_mask

        # If using token features, follows the rest
        if tokens is None:
            tokens = self.nlp(sent)
        ex = dict()
        ex['context'] = [t.text.lower() for t in tokens]
        ex['pos'] = [t.pos_ for t in tokens]
        ex['ner'] = [t.ent_type_ for t in tokens]

        ft_len = len(self.model.feature_dict)
        if self.use_idf:
            ft_len += 1
        ex_len = len(ex['context'])

        # Index words
        c_text = [self.model.word_dict[w] for w in ex['context']]

        x1 = torch.LongTensor(c_text).unsqueeze(0)
        x1_f = torch.zeros(ex_len, ft_len)
        x1_mask = torch.ByteTensor(1, ex_len).fill_(0)

        # Feature POS
        for i, w in enumerate(ex['pos']):
            if 'pos='+w in self.model.feature_dict:
                x1_f[i][self.model.feature_dict['pos='+w]] = 1.0

        # Feature NER
        for i, w in enumerate(ex['ner']):
            if 'ner='+w in self.model.feature_dict:
                x1_f[i][self.model.feature_dict['ner='+w]] = 1.0

        if self.use_idf:
            for i, w in enumerate(ex['context']):
                try:
                    x1_f[i][-1] = self.idf[w.lower()]
                except KeyError:
                    x1_f[i][-1] = 0  # ignore the tokens that are not indexed
        x1_f = x1_f.unsqueeze(0)
        return x1, x1_f, x1_mask

    def _build_ex(self, context, tokens=None):
        """Essentially using the same process of building datasets (utils.py
        and prepare_dataset.py) """
        x2, x2_f, x2_mask = self._encode_ex(context, tokens)
        return self.q_ex, self.q_f, self.q_mask, x2, x2_f, x2_mask



