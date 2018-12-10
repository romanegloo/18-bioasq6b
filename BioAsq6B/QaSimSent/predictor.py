#!/usr/bin/env python3
"""QA_Proximity predictor; Classifies if the given text is relevant to the
question."""

import logging, coloredlogs
import spacy
from spacy.tokenizer import Tokenizer
import torch
from torch.autograd import Variable
import re
import pickle
import prettytable

from .model import QaSimSent
from .. import PATHS

logger = logging.getLogger()
coloredlogs.install(
    level='DEBUG',
    fmt="[%(asctime)s %(levelname)s] %(message)s"
)


class Predictor(object):
    """Interface for computing QASim scores"""
    def __init__(self, args, nlp=None):
        """Set default properties and load a pretrained model"""
        logger.info('Initializing QaSim model...')
        self.args = args
        self.nlp = nlp if nlp is not None else spacy.load('en')
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.model, _ = QaSimSent.load(PATHS['qasim_model'])
        self.conf = self.model.conf
        self.add_words = set()
        # Model setup according to the trained model configuration
        if 'idf' in self.conf['features']:
            logger.info('Loading idf file...')
            self.idf = pickle.load(open(PATHS['idf_file'], 'rb'))
        self.q_ex = self.q_f = self.q_type = self.q_mask = None

    def get_qasim_scores(self, q, qtype, a):
        print("[Questions: {}]".format(q))
        q = self.sanitize(q)
        a = self.sanitize(a)  # This can be multiple sentences document

        # Encode the question
        self.set_q(q, qtype)
        # Batchify the candidate answers
        batch_a, sentences_a = self.batchify_context(a)
        scores = self.model.predict(batch_a)

        table = prettytable.PrettyTable(['Score', 'Sentence'])
        table.align['Score'] = 'r'
        table.align['Sentence'] = 'l'
        table.max_width['Sentence'] = 80
        for i, span in enumerate(sentences_a):
            table.add_row([scores[i].data[0], span.text])
        print(table.get_string())
        return scores, sentences_a

    def update_word_dict(self, unseen_words):
        """If new tokens exist, update the word_dict and save"""
        if len(unseen_words) > 0:
            logger.info('{} new words found. '.format(len(unseen_words)))
            logger.info('Updating word_embeddings of the Qasim model')
            self.model.update_embeddings(unseen_words, PATHS['embedding_file'])
            logger.info('After size {}'
                        ''.format(self.model.network.encoder.weight.data.shape))
            self.model.save(PATHS['qasim_model'])

    def predict_prob_b(self, body, title, docid=None):
        """Feed in to the model and get scores in batch"""
        if any(e is None
               for e in [self.q_ex, self.q_f, self.q_type, self.q_mask]):
            return [], []
        batch_t, sentences_t = self.batchify_context(self.sanitize(title))
        if batch_t is None:
            pred_t = Variable(torch.FloatTensor([-float('inf')]))
        else:
            pred_t = self.model.predict(batch_t).max()

        batch_b, sentences_b = self.batchify_context(self.sanitize(body))
        if batch_b is None:
            logger.warning("could not encode text: {}".format(body))
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
        self.q_ex, self.q_f, self.q_type, self.q_mask = self._encode_q(q, qtype)

    def _encode_q(self, q, qtype):
        tokens = self.nlp(q)
        ex = dict()
        ex['body'] = [t.text.lower() for t in tokens]
        ex['pos'] = [t.pos_ for t in tokens]
        ex['ner'] = [t.ent_type_ for t in tokens]
        ex['embedding'] = []
        for t in ex['body']:
            if t in self.model.word_dict:
                ex['embedding'].append(self.model.word_dict[t])
            else:
                ex['embedding'].append(1)
        x2 = torch.LongTensor(ex['embedding'])
        x2_f = \
            torch.FloatTensor(len(tokens), self.conf['num-features']).fill_(0)
        if 'pos' in self.model.conf['features']:
            # Feature POS
            for i, w in enumerate(ex['pos']):
                if 'pos='+w in self.model.feature_dict:
                    x2_f[i][self.model.feature_dict['pos='+w]] = 1.0

        if 'ner' in self.model.conf['features']:
            # Feature NER
            for i, w in enumerate(ex['ner']):
                if 'ner='+w in self.model.feature_dict:
                    x2_f[i][self.model.feature_dict['ner='+w]] = 1.0

        if 'idf' in self.model.conf['features']:
            if 'idf' in self.conf['features']:
                for i, w in enumerate(ex['body']):
                    try:
                        x2_f[i][-1] = self.idf[w.lower()]
                    except KeyError:
                        x2_f[i][-1] = 0  # ignore the tokens that are not indexed
        question_types = ['yesno', 'factoid', 'list', 'summary']
        x2_qtype = torch.zeros(len(question_types))
        try:
            x2_qtype[question_types.index(qtype)] = 1
        except ValueError:
            x2_qtype[3] = 1
        q_mask = torch.ByteTensor(len(tokens)).zero_()
        return x2, x2_f, x2_qtype, q_mask

    def batchify_context(self, context):
        if len(context) == 0:
            return None, []
        try:
            doc = self.nlp(context)
        except:  # SpaCy tokenizer has some issues with certain characters
            return None, []
        sentences = list(doc.sents)
        batch_len = len(sentences)
        max_doc_length = max([len(sent) for sent in sentences] + [0])
        ft_size = self.conf['num-features']

        x1 = torch.LongTensor(batch_len, max_doc_length).zero_()
        x1_mask = torch.ByteTensor(batch_len, max_doc_length).fill_(1)
        x1_f = x2_f = None
        if ft_size > 0:
            x1_f = \
                torch.FloatTensor(batch_len, max_doc_length, ft_size).zero_()
            x2_f = torch.FloatTensor(batch_len, len(self.q_ex), ft_size).zero_()
        x2 = torch.LongTensor(batch_len, len(self.q_ex)).zero_()
        x2_mask = torch.ByteTensor(batch_len, len(self.q_ex)).zero_()
        x2_qtype = torch.FloatTensor(batch_len, 4)

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
            if ft_size > 0:
                x1_f[i, :clen].copy_(x1_f_)
                x2_f[i, :, :].copy_(self.q_f)
            x2[i].copy_(self.q_ex)
            x2_qtype[i].copy_(self.q_type)

        inputs = (x1, x1_f, x1_mask, x2, x2_f, x2_qtype, x2_mask)
        return inputs, sentences

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

