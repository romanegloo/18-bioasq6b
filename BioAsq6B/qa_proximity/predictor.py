#!/usr/bin/env python3
"""QA_Proximity predictor; Classifies if the given text is relevant to the
question."""

import logging
from spacy.tokenizer import Tokenizer
import torch
import re
from .model import QaSimSent
from .. import PATHS

logger = logging.getLogger()


class Predictor(object):
    """Load a pretrained model and predict inputs"""
    def __init__(self, args, nlp=None):
        logger.info('Initializing QaSim model...')
        self.args = args
        self.model, _ = QaSimSent.load(PATHS['qasim_model'])
        self.conf = self.model.conf
        self.add_words = set()
        # The model params may not be consistent with the user given params
        self.use_idf = ('idf' in self.conf['features'])
        self.nlp = nlp
        self.tokenizer = None if self.nlp is None else Tokenizer(self.nlp.vocab)
        self.idf = None  # idf will be loaded upon the model's configuration

    def update_word_dict(self, unseen_words):
        """If new tokens exist, update the word_dict and save"""
        if len(unseen_words) > 0:
            logger.info('{} new words found. '.format(len(unseen_words)))
            logger.info('Updating word_embeddings of the Qasim model')
            self.model.update_embeddings(unseen_words, PATHS['embedding_file'])
            logger.info('After size {}'
                        ''.format(self.model.network.encoder.weight.data.shape))
            self.model.save(PATHS['qasim_model'])

    def predict_prob_b(self, body, title, docid=None, question=None,
                       qtype=None):
        """Feed in to the model and get scores in batch"""
        body = self.sanitize(body)
        title = self.sanitize(title)
        if question:
            self.set_q(question, qtype)
        batch_t, sentences_t = self._build_ex(title)
        pred_t = self.model.predict(batch_t, max_score=True)
        batch_b, sentences_b = self._build_ex(body)
        pred_b = self.model.predict(batch_b)
        # res = F.sigmoid(torch.cat((pred_t, pred_b), dim=0)).data.tolist()
        res = torch.cat((pred_t, pred_b), dim=0).data.tolist()
        snippets = []
        tmpl_ = "http://www.ncbi.nlm.nih.gov/pubmed/{}"
        # From a title
        title_text = ' '.join([s.text for s in sentences_t])
        entry = ({
            'document': tmpl_.format(docid),
            'text': title_text,
            'offsetInBeginSection': 0,
            'offsetInEndSection': len(title_text),
            'beginSection': 'title',
            'endSection': 'title'
        }, res[0])
        snippets.append(entry)
        # From body
        for i, sent in enumerate(sentences_b):
            entry = ({
                'document': tmpl_.format(docid),
                'text': sent.text,
                'offsetInBeginSection': 0,
                'offsetInEndSection': 0,
                'beginSection': '',
                'endSection': ''
            }, res[i])
            doc = body
            entry[0]['beginSection'] = 'abstract'
            entry[0]['endSection'] = 'abstract'
            offset_start = doc.find(sent.text)
            if offset_start >= 0:
                entry[0]['offsetInBeginSection'] = offset_start
                entry[0]['offsetInEndSection'] = offset_start + len(sent.text)
            snippets.append(entry)
        return res, snippets

    def set_q(self, q, qtype):
        self.q_ex, self.q_f, self.q_mask = self._encode_q(q, qtype)

    def _encode_q(self, q, qtype):
        tokens = [t.text.lower() for t in self.tokenizer(q)]
        q_idx = []
        for w in tokens:
            if w in self.model.word_dict:
                q_idx.append(self.model.word_dict[w])
            else:
                q_idx.append(1)
                self.add_words.add(w)
        question = torch.LongTensor(q_idx)
        question_types = ['yesno', 'factoid', 'list', 'summary']
        feat_q = torch.zeros(len(question_types))
        feat_q[question_types.index(qtype)] = 1
        q_mask = torch.ByteTensor(len(tokens)).zero_()
        return question, feat_q, q_mask

    def _encode_ex(self, sent, tokens=None):
        if len(self.conf['features']) == 0:
            """Run tokenizer only"""
            if tokens is None:
                tokens = self.tokenizer(sent)
            ex = dict()
            ex['context'] = [t.text.lower() for t in tokens]
            c_text = [self.model.word_dict[w] for w in ex['context']]
            x1 = torch.LongTensor(c_text).unsqueeze(0)
            x1_f = None
            x1_mask = torch.ByteTensor(1, len(ex['context'])).fill_(0)
            return x1, x1_f, x1_mask
        tokens = self.nlp(sent)
        ex = dict()
        ex['context'] = [t.text.lower() for t in tokens]
        ex['pos'] = [t.pos_ for t in tokens]
        ex['ner'] = [t.ent_type_ for t in tokens]

        ft_len = self.conf['num-features']
        ex_len = len(ex['context'])

        # Index words
        c_text = []
        for w in ex['context']:
            if w in self.model.word_dict:
                c_text.append(self.model.word_dict[w])
            else:
                c_text.append(1)
                self.add_words.add(w)
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

        if 'idf' in self.conf['features']:
            for i, w in enumerate(ex['context']):
                try:
                    x1_f[i][-1] = self.idf[w.lower()]
                except KeyError:
                    x1_f[i][-1] = 0  # ignore the tokens that are not indexed
        return x1, x1_f, x1_mask

    def _build_ex(self, context):
        """batcify"""
        if len(context) == 0:
            return None, []
        try:
            doc = self.nlp(context)
        except:  # SpaCy tokenizer has some issues with certain characters
            return None, []
        sentences = list(doc.sents)
        batch_len = len(sentences)
        max_doc_length = max([len(sent) for sent in sentences] + [0])
        ft_c_size = self.conf['num-features']
        ft_q_size = 4

        x1 = torch.LongTensor(batch_len, max_doc_length).zero_()
        x1_mask = torch.ByteTensor(batch_len, max_doc_length).fill_(1)
        x1_f = None
        if ft_c_size > 0:
            x1_f = \
                torch.FloatTensor(batch_len, max_doc_length, ft_c_size).zero_()
        x2 = torch.LongTensor(batch_len, len(self.q_ex)).zero_()
        x2_mask = torch.ByteTensor(batch_len, len(self.q_ex)).zero_()
        x2_f = torch.FloatTensor(batch_len, ft_q_size).zero_()

        for i, sent in enumerate(sentences):
            x1_, x1_f_, x1_mask_ = \
                self._encode_ex(sent.text, doc[sent.start:sent.end])
            clen = x1_.size(1)
            # Rarely the sizes do not match. Need to investigate further
            try:
                x1[i, :clen].copy_(x1_.view_as(x1[i, :clen]))
            except:
                logger.error(sent)
                raise
            x1_mask[i, :clen].fill_(0)
            if ft_c_size > 0:
                x1_f[i, :clen].copy_(x1_f_)
            x2[i].copy_(self.q_ex)
            x2_f[i, :ft_q_size].copy_(self.q_f)

        inputs = (x1, x1_f, x1_mask, x2, x2_f, x2_mask)
        return inputs, sentences

    def sanitize(self, text):
        text = re.sub(r'[<>]', ' ', text)
        return text

