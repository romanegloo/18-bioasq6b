#!/usr/bin/env python3

"""Document Rankder using Galago on medline index"""
import logging
import subprocess
import json
import tempfile
import sqlite3
import os
from collections import OrderedDict

from . import DEFAULTS
from . import utils

logger = logging.getLogger(__name__)


class GalagoRanker(object):
    def __init__(self, args, ngrams=2):
        self.args = args
        self.index_path = args.index_path
        self.db_path = args.database
        self.tokenizer = DEFAULTS['tokenizer']()
        self.ngrams = ngrams

    def batch_closest_docs(self, queries, k=10):
        """parse batch queries, run doc ranker, and return the list of ranked
        documents in trec-eval format"""
        if self.args.query_model == 'sdm':
            q_obj = self.query_sdm(queries, k)
        elif self.args.query_model == 'rm':
            q_obj = self.query_rm(queries, k)
        else:
            q_obj = self.query_baseline(queries, k)

        # save temporary query_file for galago use
        (_, fp) = tempfile.mkstemp()
        with open(fp, 'w') as out_f:
            json.dump(q_obj, fp=out_f, indent=4, separators=(',', ': '))

        # run galago
        p = subprocess.run(['galago', 'batch-search', fp],
                           stdout=subprocess.PIPE)
        if os.path.exists(fp):
            os.remove(fp)

        if self.args.verbose:
            print(p.stdout.decode('utf-8'))
        return self._batch_extract_id_scores(p.stdout.decode('utf-8'))

    def closest_docs(self, query, k=10):
        """Closest docs for one query"""
        if self.args.query_model == 'sdm':
            q_obj = self.query_sdm([query], k)
        elif self.args.query_model == 'rm':
            q_obj = self.query_rm([query], k)
        else:
            q_obj = self.query_baseline([query], k)

        # save temporary query_file for galago use
        (_, fp) = tempfile.mkstemp()
        with open(fp, 'w') as out_f:
            json.dump(q_obj, fp=out_f, indent=4, separators=(',', ': '))

        # run galago
        # todo. need to handle failure cases better
        p = subprocess.run(['galago', 'batch-search', fp],
                           stdout=subprocess.PIPE)
        if os.path.exists(fp):
            os.remove(fp)

        if self.args.verbose:
            print(p.stdout.decode('utf-8'))
        return self._extract_id_scores(p.stdout.decode('utf-8'))

    def query_sdm(self, queries, k=1):
        q_tmpl = {
            'verbose': self.args.verbose,
            'casefold': False,
            'requested': k,
            'defaultTextPart': 'postings',
            'index': self.index_path,
            'queries': []
        }
        for i, q in enumerate(queries):
            query = {'number': q['id']}
            tokens = self.tokenizer.tokenize(q['body'])
            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=1, uncased=True,
                                   filter_fn=utils.filter_ngram)
            # sdm component
            sdm_ = '#sdm({})'.format(' '.join(ngrams))
            # mesh
            ui, desc = self._mesh_ui(q['id'])
            # mesh_desc
            desc_ = []
            for t in desc:
                if len(t.split()) > 1:
                    desc_.append("#inside(#od:1({}) #field:mesh_desc())"
                                 "".format(t))
                else:
                    desc_.append("#inside({} #field:mesh_desc())".format(t))
            desc_ = "#combine({})".format(' '.join(desc_)) if len(desc_) else ''
            # mesh_ui
            ui_ = ["#inside({} #field:mesh_ui())".format(t) for t in ui]
            ui_ = "#combine({})".format(' '.join(ui_)) if len(ui_) else ''

            # combine
            q_ = "#wsum:0=5:1=1:2=4:w=1({} {} {}".format(sdm_, desc_, ui_)
            query['text'] = q_
            query['sdm.od.width'] = 2
            q_tmpl['queries'].append(query)
        return q_tmpl

    def query_rm(self, queries, k=1):
        """Pseudo Relevance Model for Semi-structured Data"""
        q_tmpl = {
            'verbose': self.args.verbose,
            'casefold': False,
            'index': self.index_path,
            'defaultTextPart': 'postings',
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

    def query_baseline(self, queries, k=1):
        q_tmpl = {
            'verbose': self.args.verbose,
            'casefold': False,  # todo, query string needs to be case sensitive
            'requested': k,
            'index': self.index_path,
            'queries': []
        }
        for i, q in enumerate(queries):
            query = {'number': 'q' + str(i)}
            tokens = self.tokenizer.tokenize(q['body'])
            # when using sdm or fdm, n-gram (n > 1) tokenizing is unnecessary
            ngrams = tokens.ngrams(n=self.ngrams, uncased=True,
                                   filter_fn=utils.filter_ngram)
            terms = []
            for t in ngrams:
                if len(t.split()) > 1:
                    terms.append('#od:2({})'.format(t))
                else:
                    terms.append(t)
            terms.extend(self._mesh_ui(q['id'])[0])

            q_ = ' '.join(terms)
            query['text'] = '#sdm({})'.format(q_)
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
        mesh_ui = []
        mesh_desc = []
        # initialize concepts db
        # todo. We are assuming that concepts db is prepared. In a real test
        # case, concepts need to be obtained adaptively.
        cnx = sqlite3.connect(self.db_path)
        csr = cnx.cursor()
        sql = "SELECT * FROM concepts WHERE id='%s';"
        for rec in csr.execute(sql % qid):
            # MetaMap tags
            if rec[1] == 'MetaMap':
                mesh_ui = rec[2].split(';')
                mesh_desc = rec[3].split(';')
            if rec[1].startswith('TmTag'):
                for idx, tag in enumerate(rec[2].split(';')):
                    elms = tag.split(':')
                    if len(elms) == 3 and elms[1] == 'MESH':
                        if elms[2] not in mesh_ui:
                            mesh_ui.append(elms[2])
                            mesh_desc.append(rec[3].split(';')[idx])

        # append TmTags
        return mesh_ui, mesh_desc

