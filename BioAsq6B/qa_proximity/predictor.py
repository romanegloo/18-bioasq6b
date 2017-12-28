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
    def __init__(self, args):
        logger.info('Initializing model...')
        if args.qaprox_model is None:
            raise RuntimeError('path to qa_prox model is required.'
                               ' (--qaprox-model)')
        self.model = QaProx.load(args.qaprox_model)
        self.nlp = spacy.load('en')

    def predict_prob(self, question, context):
        ex = self._build_ex(question, context)
        pred = self.model.predict(ex)
        return F.softmax(pred, 1).data.squeeze()

    def predict(self, question, context):
        ex = self._build_ex(question, context)
        pred = self.model.predict(ex)
        print('{:.2f} percent relevant'
              ''.format(F.softmax(pred, 1).data.squeeze()[1] * 100))

    def _build_ex(self, question, context):
        """Essentially using the same process of building datasets (utils.py
        and prepare_dataset.py) """
        q_ = self.nlp(question)
        c_ = self.nlp(context)

        ex = {}
        ex['context'] = [t.text.lower() for t in c_]
        ex['question'] = [t.text.lower() for t in q_]
        ex['pos'] = [t.pos_ for t in c_]
        ex['q_pos'] = [t.pos_ for t in q_]
        ex['ner'] = [t.ent_type_ for t in c_]
        ex['q_ner'] = [t.ent_type_ for t in q_]

        ft_len = len(self.model.feature_dict)
        x1_len = len(ex['context'])
        x2_len = len(ex['question'])

        # Index words
        c_text = [self.model.word_dict[w] for w in ex['context']]
        x1 = torch.LongTensor(c_text).unsqueeze(0)
        q_text = [self.model.word_dict[w] for w in ex['question']]
        x2 = torch.LongTensor(q_text).unsqueeze(0)
        x1_f = torch.zeros(x1_len, ft_len)
        x2_f = torch.zeros(x2_len, ft_len)
        x1_mask = torch.ByteTensor(1, x1_len).fill_(0)
        x2_mask = torch.ByteTensor(1, x2_len).fill_(0)

        # Feature POS
        for pos in ['pos', 'q_pos']:
            for i, w in enumerate(ex[pos]):
                if 'pos='+w in self.model.feature_dict:
                    if pos == 'pos':
                        x1_f[i][self.model.feature_dict['pos='+w]] = 1.0
                    else:
                        x2_f[i][self.model.feature_dict['pos='+w]] = 1.0

        # Feature NER
        for ner in ['ner', 'q_ner']:
            for i, w in enumerate(ex[ner]):
                if 'pos='+w in self.model.feature_dict:
                    if ner == 'ner':
                        x1_f[i][self.model.feature_dict['ner='+w]] = 1.0
                    else:
                        x2_f[i][self.model.feature_dict['ner='+w]] = 1.0
        x1_f = x1_f.unsqueeze(0)
        x2_f = x2_f.unsqueeze(0)

        return x1, x1_f, x1_mask, x2, x2_f, x2_mask



