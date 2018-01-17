#!/usr/bin/env python3
"""QA_Proximity predictor; Classifies if the given text is relevant to the
question."""

import logging
import spacy
import torch
import torch.nn.functional as F
from .model import QaProx

logger = logging.getLogger(__name__)


class Predictor(object):
    """Load a pretrained model and predict inputs"""
    def __init__(self, args, q=None):
        logger.info('Initializing model...')
        if args.qaprox_model is None:
            raise RuntimeError('path to qa_prox model is required.'
                               ' (--qaprox-model)')
        self.model = QaProx.load(args.qaprox_model)
        self.nlp = spacy.load('en')

    def _set_q(self, q):
        self.q_ex, self.q_f, self.q_mask = self._encode_ex(q)

    def predict_prob(self, context, question=None):
        if question:
            self._set_q(question)
        ex = self._build_ex(context)
        pred = self.model.predict(ex)
        return F.softmax(pred, 1).data.squeeze()

    # def predict(self, question, context):
    #     ex = self._build_ex(question, context)
    #     pred = self.model.predict(ex)
    #     print('{:.2f} percent relevant'
    #           ''.format(F.softmax(pred, 1).data.squeeze()[1] * 100))

    def _encode_ex(self, sent):
        s_ = self.nlp(sent)
        ex = dict()
        ex['context'] = [t.text.lower() for t in s_]
        ex['pos'] = [t.pos_ for t in s_]
        ex['ner'] = [t.ent_type_ for t in s_]

        ft_len = len(self.model.feature_dict)
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
        x1_f = x1_f.unsqueeze(0)
        return x1, x1_f, x1_mask

    def _build_ex(self, context):
        """Essentially using the same process of building datasets (utils.py
        and prepare_dataset.py) """
        x2, x2_f, x2_mask = self._encode_ex(context)
        return self.q_ex, self.q_f, self.q_mask, x2, x2_f, x2_mask



