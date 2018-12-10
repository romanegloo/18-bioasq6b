#!/usr/bin/env python3

"""Document Rankder using Galago on medline index"""
import logging
import subprocess
import json
import tempfile
import unicodedata
import pymysql
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
        self.cnx = None

    def db_connect(self):
        if self.cnx is not None and self.cnx.open:
            return
        # Read DB credential
        try:
            with open(PATHS['mysqldb_cred_file']) as f:
                host, user, passwd = f.readline().strip().split(',')
            self.cnx = pymysql.connect(
                host=host, user=user, password=passwd, db='jno236_ir',
                charset='utf8', cursorclass=pymysql.cursors.DictCursor
            )
        except (pymysql.err.DatabaseError,
                pymysql.err.IntegrityError,
                pymysql.err.MySQLError) as exception:
            logger.error('DB connection failed: {}'.format(exception))
            raise
        finally:
            if self.cnx is None:
                logger.error('Problem connecting to database')
            else:
                logger.debug('jno236_ir connected')

    def db_close(self):
        if self.cnx and self.cnx.open:
            self.cnx.close()

    def closest_docs(self, query, k=10):
        """Return RankedDocs of k closest docs for the query"""
        ranked_docs = RankedDocs(query)
        # Save temporary query_file for galago use
        g_verbose = (self.args.verbose and self.args.qid is not None)
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
            return ranked_docs
        elif self.args.query_model in ['sdm', 'sdm_mesh', 'rm3']:
            if self.args.query_model == 'sdm':
                query_fn = self.query_sdm
            elif self.args.query_model == 'sdm_mesh':
                query_fn = self.query_sdm_mesh
            elif self.args.query_model == 'rm3':
                query_fn = self.query_rm
            use_stemmer = True  # By default, use stemmer
            q_tmpl = {
                'verbose': g_verbose,
                'casefold': False,
                'requested': k,
                'defaultTextPart': 'postings.krovetz',
                'index': self.index_path,
                'queries': []
            }
            q_obj = query_fn([query], q_tmpl=q_tmpl)
            # Check if to use Krovetz stemmer
            try:
                tokens = [d.text for d in self.nlp(query['body'])]
            except KeyError:
                logger.warning('spacy error. normalizing...')
                qbody_norm = unicodedata.normalize('NFD', query['body'])
                tokens = [d.text for d in self.nlp(qbody_norm)]

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
                q_obj = query_fn([query], q_tmpl=q_tmpl)
            (_, fp) = tempfile.mkstemp()
            json.dump(q_obj, fp=open(fp, 'w'), indent=4, separators=(',', ': '))
            p = subprocess.run(['galago', 'batch-search', fp],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if g_verbose:
                logger.info(p.stdout.decode('utf-8'))
            docids_scores = self._extract_id_scores(p.stdout.decode('utf-8'))
            ranked_docs.rankings, ranked_docs.scores['retrieval'] = docids_scores
        return ranked_docs

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
            terms = list(set(self.sanitize(terms, 'ngrams')))
            for t in ngrams:
                if len(t.split()) > 1:
                    terms.append('#uw:2({})'.format(
                        ' '.join(self.sanitize(t.split(), 'ngrams'))))
            combine_ = '#combine({})'.format(' '.join(terms))
            query['text'] = combine_
            q_tmpl['queries'].append(query)
        return q_tmpl

    def query_rm(self, queries, q_tmpl):
        """RM3 model is default"""
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            tokens = self.tokenizer.tokenize(q['body'])
            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=1, uncased=True,
                                   filter_fn=utils.filter_ngram)
            # sdm component
            rm_ = '#rm({})'.format(' '.join(self.sanitize(ngrams, 'ngrams')))
            query['text'] = rm_
            query['fbDocs'] = 10
            query['fbTerm'] = 5
            query['fbOrigWeight'] = 0.75
            q_tmpl['queries'].append(query)
        return q_tmpl

    def query_sdm(self, queries, q_tmpl):
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            tokens = self.tokenizer.tokenize(q['body'])
            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=1, uncased=True,
                                   filter_fn=utils.filter_ngram)
            # sdm component
            sdm_ = '#sdm({})'.format(' '.join(self.sanitize(ngrams, 'ngrams')))
            query['text'] = sdm_
            query['sdm.od.width'] = 3
            q_tmpl['queries'].append(query)
        return q_tmpl

    def query_sdm_mesh(self, queries, q_tmpl):
        """query: #sdm() ().mesh_ui ().chemical_ui"""
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            ui, desc = self._mesh_ui(q['id'])
            str_ui = ' '.join(ui)
            tokens = self.tokenizer.tokenize(q['body'])
            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=1, uncased=True,
                                   filter_fn=utils.filter_ngram)
            # sdm component
            sdm_ = '#sdm({})'.format(' '.join(self.sanitize(ngrams, 'ngrams')))
            query['text'] = sdm_
            if len(ui) > 0:
                query['text'] = \
                    '#combine:0=0.9:1=0.1({} #combine({}))'.format(sdm_, str_ui)
            query['sdm.od.width'] = 3
            q_tmpl['queries'].append(query)
        return q_tmpl

    def sanitize(self, tokens, type):
        if type == 'band':
            tokens_ = []
            for t in tokens:
                t = re.sub('[.\?]', ' ', t)
                t = re.sub('[,;()]', '', t)
                if len(t.split()) > 1:
                    tokens_.extend(t.split())
                else:
                    tokens_.append(t)
            return tokens_
        if type == 'ngrams':
            tokens = [re.sub('[?\']', '', t) for t in tokens]
            tokens = [re.sub('[.;()/]', ' ', t) for t in tokens]
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
        self.db_connect()
        with self.cnx.cursor() as cursor:
            sql = "SELECT * FROM BIOASQ_CONCEPT WHERE id=%s;"
            cursor.execute(sql, (qid, ))
            for rec in cursor.fetchall():
                # Add MTI MeSH
                if rec['source'] == 'MTI (MeSH)' and rec['cid'] not in mesh_ui:
                    mesh_ui.add(rec['cid'])
                    mesh_desc.add(rec['name_0'])
                # Add TmTags MESH
                if rec['source'].startswith('TmTag'):
                    for i, tag in enumerate(rec['cid'].split(';')):
                        elms = tag.split(':')
                        if len(elms) == 3 and elms[1] == 'MESH' and \
                                elms[2] not in mesh_ui:
                            mesh_ui.add(elms[2])
                            mesh_desc.add(rec['name_0'].split(';')[i])
        return mesh_ui, mesh_desc
