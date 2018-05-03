#!/usr/bin/env python3

"""Document Rankder using Galago on medline index"""
import logging
import subprocess
import json
import tempfile
import sqlite3
from collections import OrderedDict
import re
import krovetzstemmer

from . import DEFAULTS, utils
from .. import PATHS
from ..common import AverageMeter, RankedDocs

logger = logging.getLogger()


class GalagoSearch(object):
    def __init__(self, args, nlp=None, idf=None):
        self.args = args
        self.index_path = PATHS['galago_idx']
        self.db_path = PATHS['concepts_db']
        self.tokenizer = DEFAULTS['tokenizer']()
        self.ngrams = 2
        self.use_stemmer_threshold = 3  # idf ratio of stemmed/non-stemmed token
        self.nlp = nlp
        self.idf = idf
        self.stemmer = krovetzstemmer.Stemmer()

    def closest_docs(self, query, k=10, cache=None):
        """Return RankedDocs of k closest docs for the query"""
        ranked_docs = RankedDocs(query)
        # Read from cached scores, if cache is given
        if cache is not None:
            # Use the scores only when it has scores >= the requested number
            if len(cache['rankings']) >= min(len(ranked_docs.rankings), k):
                ranked_docs.rankings = cache['rankings'][:k]
                ranked_docs.scores['retrieval'] = cache['scores-ret'][:k]
                return ranked_docs
        # Save temporary query_file for galago use
        g_verbose = (self.args.qid is not None)
        if self.args.query_model == 'baseline':
            q_tmpl = {
                'verbose': g_verbose,
                'casefold': False,
                'requested': k,
                'defaultTextPart': 'postings',
                'index': self.index_path,
                'queries': []
            }
            q_obj = self.query_baseline([query], q_tmpl=q_tmpl)
            (_, fp) = tempfile.mkstemp()
            json.dump(q_obj, fp=open(fp, 'w'), indent=4, separators=(',', ': '))
            p = subprocess.run(['galago', 'batch-search', fp],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if g_verbose:
                logger.info(p.stdout.decode('utf-8'))
            ranked_docs.rankings, ranked_docs.scores['retrieval'] = \
                self._extract_id_scores(p.stdout.decode('utf-8'))
            ranked_docs.update_cache.append('retrieval')
            return ranked_docs
        elif self.args.query_model == 'sdm':
            use_stemmer = True  # By default, use stemmer
            q_tmpl = {
                'verbose': g_verbose,
                'casefold': False,
                'requested': k,
                'defaultTextPart': 'postings.krovetz',
                'index': self.index_path,
                'queries': []
            }
            q_obj = self.query_sdm([query], q_tmpl=q_tmpl)
            # Check if to use Krovetz stemmer
            tokens = [d.text for d in self.nlp(query['body'])]
            for t in tokens:
                try:
                    ratio = self.idf[t.lower()] / self.idf[self.stemmer.stem(t)]
                except KeyError:
                    ratio = 1
                if ratio > self.use_stemmer_threshold:
                    logger.info("Not using stemmer: {}".format(t))
                    use_stemmer = False
                    break
            if not use_stemmer:
                q_tmpl['defaultTextPart'] = 'postings'
                q_tmpl['queries'] = []
                q_obj = self.query_sdm([query], q_tmpl=q_tmpl)
            (_, fp) = tempfile.mkstemp()
            json.dump(q_obj, fp=open(fp, 'w'), indent=4, separators=(',', ': '))
            logger.info('galago search {}'.format(query['id']))
            p = subprocess.run(['galago', 'batch-search', fp],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            logger.info('search done')
            if g_verbose:
                logger.info(p.stdout.decode('utf-8'))
            docids_scores = self._extract_id_scores(p.stdout.decode('utf-8'))
            ranked_docs.rankings, ranked_docs.scores['retrieval'] = docids_scores
        ranked_docs.update_cache.append('retrieval')
        return ranked_docs

    def sanitize(self, tokens, type):
        if type == 'band':
            tokens_ = []
            for t in tokens:
                t = re.sub('[.?]', ' ', t)
                t = re.sub('[,;()]', '', t)
                if len(t.split()) > 1:
                    tokens_.extend(t.split())
                else:
                    tokens_.append(t)
            return tokens_
        if type == 'ngrams':
            tokens = [re.sub('[\']', '', t) for t in tokens]
            tokens = [re.sub('[.;()?/]', ' ', t) for t in tokens]
            # To query for a hyphenated term; Galago tokenizes on hyphens
            tokens = ["#od:1({})".format(re.sub('[-]', ' ', t))
                      if ('-' in t) else t for t in tokens]
            return tokens
        if type == 'mesh_desc':
            # remove auxiliary information in a parenthesis
            tokens = [re.sub(r"\(.*\)", '', t) for t in tokens]
            # remove non-alphanumeric characters
            tokens = [re.sub('[.,;()]', '', t) for t in tokens]
            tokens = [t for t in tokens]
            # hyphens
            tokens = ["#od:1({})".format(re.sub('[-]', ' ', t))
                      if ('-' in t) else t for t in tokens]
            return tokens
        if type == 'mesh_ui':
            tokens = [t for t in tokens]
            return tokens

        return tokens

    def query_sdm(self, queries, q_tmpl):
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            tokens = self.tokenizer.tokenize(q['body'])

            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=1, uncased=True,
                                   filter_fn=utils.filter_ngram)
            band_ = '#bool(#band({}))'\
                    ''.format(' '.join(self.sanitize(ngrams, 'band')))
            # sdm component
            sdm_ = '#sdm({})'.format(' '.join(self.sanitize(ngrams, 'ngrams')))
            # mesh
            ui, desc = self._mesh_ui(q['id'])
            # mesh_desc
            desc_ = []
            desc = self.sanitize(desc, 'mesh_desc')
            for t in desc:
                if len(t.split()) > 1:
                    desc_.append('#od:2({}).mesh_desc'.format(t))
                else:
                    desc_.append('{}.mesh_desc'.format(t))
            desc_c = '#band({})'.format(' '.join(desc_)) \
                if len(desc_) else ''
            # mesh_ui
            ui = self.sanitize(ui, 'mesh_ui')
            ui_ = ['{}.mesh_ui'.format(t) for t in ui]
            ui_c = '#band({})'.format(' '.join(ui_)) if len(ui_) else ''

            # combine
            mesh_ = "#bool({} {})".format(desc_c, ui_c) \
                if (len(desc_c + ui_c) > 0) else ''
            mesh_ = "#bool({})".format(ui_c) if (len(desc_c + ui_c) > 0) else ''
            if self.args.galago_weights:
                weights_ = list(map(float, self.args.galago_weights.split(',')))
                q_ = '#wsum:0={}:1={}:w=1({} {})'\
                     ''.format(*weights_, sdm_, mesh_)
            else:
                q_ = '#wsum:0=6:1=1:w=1({} {})'.format(sdm_, mesh_)
            # query['text'] = q_
            # query['text'] = "#combine({} {})".format(band_, sdm_)
            query['text'] = sdm_
            query['sdm.od.width'] = 2
            q_tmpl['queries'].append(query)
        return q_tmpl

    def query_rm(self, queries, k=1):
        """Pseudo Relevance Model for Semi-structured Data"""
        q_tmpl = {
            'verbose': self.args.verbose,
            'casefold': False,
            'index': self.index_path,
            'defaultTextPart': 'postings.krovetz',
            'requested': k,
            'relevanceModel':
                "org.lemurproject.galago.core.retrieval.prf.RelevanceModel3",
            'fbDocs': 3,
            'fbTerm': 5,
            'fbOrigWeight': 0.75,
            'rmstopwords': 'rmstop',
            'queries': []
        }
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            tokens = self.tokenizer.tokenize(q['body'])
            ngrams = tokens.ngrams(n=1, uncased=True,
                                   filter_fn=utils.filter_ngram)
            # terms
            terms_ = ["#od:2({})".format(t) if len(t.split()) > 1 else t
                     for t in ngrams]
            # mesh
            ui, desc = self._mesh_ui(q['id'])

            q_ = "#rm({})".format(' '.join(terms_ + ui))
            query['text'] = q_
            query['sdm.od.width'] = 2
            q_tmpl['queries'].append(query)
        return q_tmpl

    def query_baseline(self, queries, q_tmpl):
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            tokens = self.tokenizer.tokenize(q['body'])
            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=3, uncased=True,
                                   filter_fn=utils.filter_ngram)
            terms = []
            for ngram in ngrams:
                terms.extend(ngram.split(' '))
            terms = list(set(terms))
            for t in ngrams:
                if len(t.split()) > 1:
                    terms.append('#uw:2({})'.format(t))
            combine_ = \
                '#combine({})'.format(' '.join(self.sanitize(terms, 'ngrams')))
            query['text'] = combine_
            q_tmpl['queries'].append(query)
        return q_tmpl

    def _batch_extract_id_scores(self, results):
        resDict = OrderedDict()
        for rec in results.splitlines():
            fields = rec.split()
            if len(fields) == 6 and fields[5] == 'galago':
                if fields[0] not in resDict:
                    resDict[fields[0]] = {'docids': [], 'scores': []}
                resDict[fields[0]]['docids'].append(
                    fields[2].replace('PMID-', ''))
                resDict[fields[0]]['scores'].append(float(fields[4]))

        lst_docids = []
        lst_scores = []
        for _, q in resDict.items():
            lst_docids.append(q['docids'])
            lst_scores.append(q['scores'])
            assert len(q['docids']) == len(q['scores']), \
                "doc_id does not map to scores exactly"
        return lst_docids, lst_scores

    def _extract_id_scores(self, results):
        doc_ids = []
        scores = []
        for rec in results.splitlines():
            fields = rec.split()
            if len(fields) == 6 and fields[5] == 'galago':
                doc_ids.append(fields[2].replace('PMID-', ''))
                scores.append(float(fields[4]))

        assert len(doc_ids) == len(scores), \
            "doc_id does not map to scores exactly"
        return doc_ids, scores

    def _mesh_ui(self, qid):
        """obtain stored mesh UIs from concept DB or run metamap to get them"""
        # just use independent lists of uis and names
        mesh_ui = set()
        mesh_desc = set()
        # initialize concepts db
        cnx = sqlite3.connect(self.db_path)
        csr = cnx.cursor()
        sql = "SELECT * FROM concepts2 WHERE id='%s';"
        for rec in csr.execute(sql % qid):
            # MetaMap tags
            if rec[1] == 'MetaMap':
                mesh_ui |= set([c.lower() for c in rec[2].split(';')])
                mesh_desc |= set([c.lower() for c in rec[3].split(';')])
            if rec[1].startswith('TmTag'):
                for idx, tag in enumerate(rec[2].split(';')):
                    elms = tag.split(':')
                    if len(elms) == 3 and elms[1] == 'MESH':
                        if elms[2] not in mesh_ui:
                            mesh_ui.add(elms[2].lower())
                            mesh_desc.add(rec[3].split(';')[idx].lower())

        # append TmTags
        return mesh_ui, mesh_desc
