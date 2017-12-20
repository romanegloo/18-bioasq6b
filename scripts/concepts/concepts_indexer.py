#!/usr/bin/env python3
"""
Usage:
    python concepts_indexer.py {tmtools,metamap,go}
        --target {pubmed,bioasq}
        --input_file [path_to_input_datafile]
        --database [path_to_output_database]

doc. Building Concepts DB (http://bit.ly/2AFob0Z)
"""
from __future__ import print_function

import argparse
import time
import requests
import os
import sys
import logging
import json
import sqlite3
from pathlib import PosixPath
import urllib.parse
from tqdm import tqdm  # progress meter
import time


DATA_DIR = \
    os.path.join(PosixPath(__file__).absolute().parents[4].as_posix(), 'data')

# initialize logger
logger = logging.getLogger()

class GoClient(object):
    def __init__(self):
        self.golr_url = "http://golr.geneontology.org/select?"
        self.params = {
            'defType': 'edismax',  # use solr's extended DisMax Query Parser
            'qt': 'standard',      # query type
            # 'indent': 'ON',        # indentation of the response
            'wt': 'json',          # specifies the response format, default: XML
            'rows': '10',          # maximum number of returned records
            'start': '0',          # paging
            'fl': 'annotation_class,description,source,idspace,synonym,'
                  'alternate_id,annotation_class_label,score,id',
                # fields to be returned
            'facet': 'true',       # arrange search results into categories
            'facet.sort': 'count',
            'facet.limit': '25',
            'facet.field': ['source', 'idspace', 'subset', 'is_obsolete'],
            'json.nl': 'arrarr',   # json NamedList
                # (ref. Solr JSON-specific parameters)
            'fq': ['document_category:"ontology_class"',  # filter query
                   'idspace:"GO"',
                   'is_obsolete:"false"'],
            'q': '',
            # Query Fields, specifies the fields in the index on which
            # to perform the query
            'qf': ['annotation_class^3',
                   'annotation_class_label_searchable^5.5',
                   'description_searchable^1',
                   'synonym_searchable^1',
                   'alternate_id^1']
        }

    def request(self, q):
        url = self.golr_url
        self.params['q'] = q
        for k, v in self.params.items():
            if type(v) is list:
                for entry in v:
                    e = urllib.parse.quote_plus(str(entry))
                    url += "{}={}&".format(k, e)
            else:
                e = urllib.parse.quote_plus(str(v))
                url += "{}={}&".format(k, e)
        r = requests.get(url)
        if r.status_code == 200:
            rst = json.loads(r.text)
            return rst
        else:
            logger.error("GO API request failed: %s" % r.text)
        return

# GoClient end
# ------------------------------------------------------------------------------


def _run_bioasq_tmtools():
    # read input file
    questions = []
    with open(args.input_file) as f:
        data = json.load(f)
        for entry in data['questions']:
            # remove escaped double quotes
            q_body = entry['body'].replace('\"', '')
            q = {
                'sourcedb': 'BioAsq',
                'sourceid': entry['id'],
                'text': q_body
            }
            questions.append(q)
    logger.info('{} questions read'.format(len(questions)))

    tmtools_triggers = {
        'Gene': 'TmTag-G',
        'Mutation': 'TmTag-V',
        'Species': 'TmTag-S',
        'Chemical': 'TmTag-C',
        'Disease': 'TmTag-D'
    }
    tmtools_url = \
        "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/"
    for trigger in tmtools_triggers.keys():
        logger.info('sending requests to TmTools... (tagger: {})'
                    ''.format(trigger))
        input_str = ','.join([json.dumps(q, separators=(',', ':'))
                              for q in questions])
        url_submit = tmtools_url + trigger + '/Submit/'
        submit_resp = requests.post(url_submit, input_str)

        sess = submit_resp.text
        logger.info("Request received (Session number: {})".format(sess))

        url_receive = tmtools_url + sess + "/Receive/"

        code = 404
        while code == 404 or code == 501:  # loop until it gets response
            time.sleep(15)
            try:
                result_resp = requests.get(url_receive)
            except requests.HTTPError as e:
                code = result_resp.status_code
                logger.warning('status: {}'.format(code))
            else:
                code = result_resp.status_code
                if code != 200:
                    logger.warning('status: {}, {}'
                                   ''.format(code, result_resp.text.strip()))
        _store_tm_results(result_resp.text, tmtools_triggers[trigger])


def _store_tm_results(resp, trigger):
    """parse TmTools results and store in db"""
    try:
        data = json.loads('[' + resp + ']')
    except:
        print(resp)
        e = sys.exc_info()[0]
        raise RuntimeError("Cannot load response in JSON: {}".format(e))

    cnx = sqlite3.connect(args.database)
    csr = cnx.cursor()
    pairs = []
    for entry in data:
        if len(entry['denotations']) > 0:
            q_body = entry['text']
            concepts = []
            for obj in entry['denotations']:
                cid = obj['obj']
                term = q_body[obj['span']['begin']: obj['span']['end']]
                term = term.replace(';', ' ')
                concepts.append((cid, term))
            cids = ';'.join([t[0] for t in concepts])
            texts = ';'.join(t[1] for t in concepts)

            pairs.append((entry['sourceid'], trigger, cids, texts))
            print("{} added\r".format(entry['sourceid']), end='')
    csr.executemany("INSERT OR REPLACE INTO concepts VALUES (?,?,?,?)", pairs)
    logger.info("{} records inserted".format(len(pairs)))
    cnx.commit()
    cnx.close()


def _run_bioasq_metamap():
    from BioAsq6b.data_services import MetamapExt

    # read input file
    logger.info('Reading BioAsq dataset input file...')
    questions = []
    with open(args.input_file) as f:
        data = json.load(f)
        for entry in data['questions']:
            # remove escaped double quotes
            q_body = entry['body'].replace('\"', '')
            q = {'qid': entry['id'], 'body': q_body}
            questions.append(q)
    logger.info('{} questions read'.format(len(questions)))

    # get MeSH terms
    logger.info('Extracting MeSH terms from the questions...')
    mm = MetamapExt()
    records = []
    for q in tqdm(questions, desc='MetaMap'):
        mesh_tags = set()
        mesh_names = set()
        rec = mm.get_mesh_descriptors(q['body'])
        for concept in rec:
            mesh_tags.add(concept[1])
            mesh_names.add(concept[2].replace(';', ''))
        records.append((q['qid'], 'MetaMap',
                       ';'.join(mesh_tags), ';'.join(mesh_names)))

    # store the results into db
    logger.info('Storing MetaMap concepts (MeSH) into db...')
    cnx = sqlite3.connect(args.database)
    csr = cnx.cursor()
    csr.executemany("INSERT OR REPLACE INTO concepts VALUES (?,?,?,?)",
                    records)
    cnx.commit()
    logger.info('{} records inserted'.format(cnx.total_changes))
    cnx.close()


def _run_pubmed():
    raise NotImplementedError


def _run_go_expand():
    """this does not require input data. It reads stored concepts,
    in particular gene, and obtain GO concepts via golr service. The results
    are used to expand the concept database."""
    # read all the records with type 'TmTag-G' and tags 'Gene:'
    logger.info('Reading concept database for gene names...')
    gene_names = dict()
    cnx = sqlite3.connect(args.database)
    csr = cnx.cursor()
    sql = """SELECT * FROM concepts WHERE type='TmTag-G' or type='TmTag-V';"""
    csr.execute(sql)
    tm_rows = csr.fetchall()
    for row in tm_rows:
        (id, type, tags, names) = row
        names = names.split(';')
        for idx, tag in enumerate(tags.split(';')):
            if tag.startswith('Gene:'):
                gene_names[names[idx]] = None

    # request
    logger.info('Sumbitting request to Golr..')
    goclient = GoClient()
    for g in tqdm(gene_names.keys(), desc="golr request"):
        go_concepts = []
        time.sleep(.5)
        rst = goclient.request(g)
        if rst:
            # process the result
            max_score = rst['response']['maxScore']
            num_found = rst['response']['numFound']
            if num_found == 0:
                continue
            docs = rst['response']['docs']
            for d in docs:
                # if d['score'] < .9 * max_score:
                #     continue  # we only consider 90% or above confidence
                go_concepts.append((d['id'], d['annotation_class_label']))
            gene_names[g] = go_concepts

    # store GO concepts into db
    logger.info('Storing GO concepts into db...')
    records = []
    for row in tqdm(tm_rows, desc="expanding concepts"):
        (id, type, tags, names) = row
        names = names.split(';')
        go_tags = []
        go_names = []
        for idx, tag in enumerate(tags.split(';')):
            if tag.startswith('Gene:') and gene_names[names[idx]]:
                for c in gene_names[names[idx]]:
                    go_tags.append(c[0])
                    go_names.append(c[1].replace(';', ''))
        if len(go_tags) > 0:
            records.append((id, 'GO', ';'.join(go_tags), ';'.join(go_names)))
    csr.executemany("INSERT OR REPLACE INTO concepts VALUES (?,?,?,?);",
                    records)
    cnx.commit()
    cnx.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str,
                        choices=['tmtools', 'metamap', 'go'],
                        help='concept sources')
    parser.add_argument('--target', type=str, choices=['pubmed', 'bioasq'],
                        default='bioasq',
                        help='either pmid list or bioasq questions')
    parser.add_argument('--input_file', type=str, help='path to the input file')
    parser.add_argument('--database', type=str,
                        help='path to the sqlite database file to store '
                             'the results')
    args = parser.parse_args()

    # set defaults
    if args.input_file is None:
        if args.target == 'bioasq':
            args.input_file = os.path.join(DATA_DIR,
                                           'bioask/BioASQ-training5b/',
                                           'BioASQ-trainingDataset5b.json')
        elif args.target == 'pubmed':
            raise RuntimeError("input file is required for pubmed indexing")
    if args.database is None:
        args.database = os.path.join(DATA_DIR, 'concepts.db')
    if not os.path.isfile(args.database):
        # create database
        logger.info('Creating a new concept database')
        cnx = sqlite3.connect(args.database)
        csr = cnx.cursor()
        sql = """CREATE TABLE IF NOT EXISTS concepts 
                (id text, type text, tags text, names text,
                 PRIMARY KEY (id, type));"""
        csr.execute(sql)

    # logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if args.target == 'bioasq':
        if args.source == 'tmtools':
            _run_bioasq_tmtools()
        elif args.source == 'metamap':
            _run_bioasq_metamap()
    elif args.source == 'pubmed':
        _run_pubmed()

    if args.source == 'go':  # args.target is not required
        _run_go_expand()


