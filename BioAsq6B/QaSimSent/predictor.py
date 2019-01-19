#!/usr/bin/env python3
"""QA_Proximity predictor; Classifies if the given text is relevant to the
question."""

from typing import Tuple, List
import logging
import spacy
from spacy.tokenizer import Tokenizer
import torch
import  torch.functional as F
import re
import pickle
import prettytable

from .model import QaSimSent
from .. import PATHS

logger = logging.getLogger()


class Predictor(object):
    """Interface for computing QASim scores"""
    def __init__(self, model_path=None, nlp=None, load_wd=False):
        """Set default properties and load a pretrained model"""
        self.nlp = nlp if nlp is not None else spacy.load('en')
        self.tokenizer = Tokenizer(self.nlp.vocab)
        if model_path is None:
            self.model, _ = QaSimSent.load(PATHS['qasim_model'])
        else:
            self.model, _ = QaSimSent.load(model_path, load_wd=load_wd)
        self.conf = self.model.conf
        self.add_words = set()
        # Model setup according to the trained model configuration
        if 'idf' in self.conf['features']:
            logger.info('Loading idf file...')
            self.idf = pickle.load(open(PATHS['idf_file'], 'rb'))
        self.q_ex = self.q_f = self.q_type = self.q_mask = None

    def get_qasim_scores(self, q, qtype, a):
        """Called by an interactive script"""
        print("[Questions: {}]".format(q))
        # Encode the question
        self.set_q(self.sanitize(q), qtype)
        # Batchify the candidate answers
        batches, doc = self.batchify(self.sanitize(a))
        if batches is None:
            return [], []
        predictions = self.model.predict(batches)

        table = prettytable.PrettyTable(['Score', 'Sentence'])
        table.align['Score'] = 'r'
        table.align['Sentence'] = 'l'
        table.max_width['Sentence'] = 80
        for i, sent in enumerate(doc.sents):
            table.add_row([predictions[i].item(), sent.text])
        print(table.get_string())
        return predictions, doc

    def predict_prob_b(self, body, docid):
        """Called by QA_reranker; Run the model on a document body (ignore the
        title, assuming that the title does not answer any question)"""
        """Feed in to the model and get scores in batch"""
        results = list()  # List of results by sentences
        # A question must be given
        if any(e is None
               for e in [self.q_ex, self.q_f, self.q_type, self.q_mask]):
            return [], []
        # Apply the model on the body document
        batch_b, doc = self.batchify(self.sanitize(body))
        if batch_b is None:
            return [], []
        pred_b = self.model.predict(batch_b)
        res = torch.sigmoid(pred_b)
        # From body
        assert len(res) == len(list(doc.sents))
        for i, sent in enumerate(doc.sents):
            entry = {
                'document': "http://www.ncbi.nlm.nih.gov/pubmed/" + docid,
                'text': sent.text,
                'offsetInBeginSection': sent.start_char,
                'offsetInEndSection': sent.end_char,
                'beginSection': 'abstract',
                'endSection': 'abstract',
                'score': res[i].item()
            }
            results.append(entry)
        return results

    def set_q(self, q, qtype):
        self.q_ex, self.q_f, self.q_type, self.q_mask = self._encode_q(q, qtype)

    def _encode_q(self, q, qtype):
        tokens = self.nlp(q)
        text_lower = [t.text.lower() for t in tokens]
        q_ = [self.model.word_dict[t] if t in self.model.word_dict else 0
              for t in text_lower]
        q = torch.LongTensor(q_)
        q_f = torch.zeros(len(tokens), self.conf['num-features'])

        if 'pos' in self.model.conf['features']:
            # Feature POS
            for i, t in enumerate(tokens):
                if 'pos=' + t.pos_ in self.model.feature_dict:
                    q_f[i][self.model.feature_dict['pos='+t.pos_]] = 1
        if 'ner' in self.model.conf['features']:
            # Feature NER
            for i, t in enumerate(tokens):
                if 'ner=' + t.ent_type_ in self.model.feature_dict:
                    q_f[i][self.model.feature_dict['ner='+t.ent_type_]] = 1
        if 'idf' in self.model.conf['features']:
            if 'idf' in self.conf['features']:
                for i, t in enumerate(text_lower):
                    try:
                        q_f[i][-1] = self.idf[t]
                    except KeyError:
                        q_f[i][-1] = 0  # ignore the tokens that are not indexed
        question_types = ['yesno', 'factoid', 'list', 'summary']
        q_type = torch.zeros(len(question_types), dtype=torch.float)
        try:
            q_type[question_types.index(qtype)] = 1
        except ValueError:
            q_type[3] = 1
        q_mask = torch.zeros(len(tokens), dtype=torch.uint8)

        return q, q_f, q_type, q_mask

    def batchify(self, context):
        if len(context) == 0:
            return None, []
        try:
            doc = self.nlp(context)
        except:  # SpaCy tokenizer has some issues with certain characters
            return None, []
        batch_len = len(list(doc.sents))
        max_doc_length = max([len(s) for s in doc.sents] + [0])
        ft_size = self.conf['num-features']

        c = torch.zeros(batch_len, max_doc_length, dtype=torch.long)
        c_mask = torch.ones(batch_len, max_doc_length, dtype=torch.uint8)
        c_f = None
        if ft_size > 0:
            c_f = torch.zeros(batch_len, max_doc_length, ft_size)

        for i, sent in enumerate(doc.sents):
            c_, c_f_, c_mask_ = \
                self._encode_ex(sent.text, doc[sent.start:sent.end])
            clen = c_.size(1)
            try:
                c[i, :clen].copy_(c_.view_as(c[i, :clen]))
            except:
                logger.error(sent)
                raise
            c_mask[i, :clen].fill_(0)
            if ft_size > 0:
                c_f[i, :clen].copy_(c_f_)

        # Repeat the question tensors
        q_ex = self.q_ex.unsqueeze(0).repeat(batch_len, 1)  # batch x qlen
        q_f = self.q_f.unsqueeze(0).repeat(batch_len, 1, 1)  # batch x qlen x nf
        q_type = self.q_type.repeat(batch_len, 1)  # batch x 4
        q_mask = self.q_mask.unsqueeze(0).repeat(batch_len, 1)  # batch x qlen
        inputs = (c, c_f, c_mask, q_ex, q_f, q_type, q_mask)
        return inputs, doc

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

    def sanitize(self, text):
        if text is None:
            return ''
        # clean up the text before using a Tokenizer
        text = re.sub('[\n?\']', '', text)
        text = re.sub('[()<>/]', ' ', text)
        text = re.sub('\s+', ' ', text)
        return text

