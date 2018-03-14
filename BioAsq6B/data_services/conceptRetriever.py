"""Full pipeline of getting a list of relevant concepts from the given
question text"""

import logging
import json
import requests
import urllib.parse
import time
import sys
from tqdm import tqdm
import sqlite3
from . import MetamapExt
from .. import PATHS

# initialize logger
logger = logging.getLogger()
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ConceptRetriever(object):
    def __init__(self, args, updateDatabase=False):
        self.args = args
        self.concepts = dict()
        self.updateDatabase = updateDatabase
        self.db = PATHS['concepts_db']

    def run_tmtools(self, questions):
        """Using TmTools to extract concepts from question texts
        (http://bit.ly/2G08JNt)"""
        q_body = []
        for q in questions:
            entry = {
                'sourcedb': 'BioAsq',
                'sourceid': q['id'],
                'text': q['body'].replace('\"', '')
            }
            q_body.append(entry)
        tmtools_triggers = {
            'Gene': 'TmTag-G',
            'Mutation': 'TmTag-V',
            'Species': 'TmTag-S',
            'Chemical': 'TmTag-C',
            'Disease': 'TmTag-D'
        }
        tmtools_url = "https://www.ncbi.nlm.nih.gov/CBBresearch/" \
                      "Lu/Demo/RESTful/tmTool.cgi/"
        for trigger in tmtools_triggers.keys():
            logger.info('Sending requests to TmTools... (tagger: {})'
                        ''.format(trigger))
            # Request
            input_str = ','.join([json.dumps(q, separators=(',', ':'))
                                  for q in q_body])
            url_submit = tmtools_url + trigger + '/Submit/'
            submit_resp = requests.post(url_submit, input_str)
            sess = submit_resp.text
            logger.info("Request received (Session number: {})".format(sess))

            # Waiting for the response
            url_receive = tmtools_url + sess + "/Receive/"
            max_timeout = 60 * 10  # 10 minutes
            elapsed = 0
            code = 404  # Initially 404
            resp = None
            while code in [404, 501]:
                time.sleep(30)
                elapsed += 30
                try:
                    resp = requests.get(url_receive)
                except requests.HTTPError as e:
                    code = resp.status_code
                    logger.warning('status: {}'.format(code))
                else:
                    code = resp.status_code
                    if code != 200:
                        print('status:{}, {} ({}) {}\r'
                              ''.format(code, resp.text.strip(), elapsed,
                                        ' ' * 20),
                              end='', flush=True)
                if elapsed > max_timeout:
                    logger.warning('TmTools timeout reached')
                    break

            # Parse the results
            try:
                data = json.loads('[' + resp.text + ']')
            except:
                logger.info(resp.text)
                e = sys.exc_info()[0]
                logger.error("Cannot load response in JSON: {}".format(e))
                return
            for q_resp in data:
                for concept in q_resp['denotations']:
                    b = concept['span']['begin']
                    e = concept['span']['end']
                    term = q_resp['text'][b:e]
                    # term = term.replace(';', ' ')
                    entry = {
                        'source': "TmTools ({})".format(trigger),
                        'cid': concept['obj'],
                        'name0': term,
                        'nameN': term
                    }
                    if q_resp['sourceid'] not in self.concepts:
                        self.concepts[q_resp['sourceid']] = []
                    self.concepts[q_resp['sourceid']].append(entry)

    def run_metamap(self, questions):
        """Runs MataMap to extract MeSH concepts"""
        # get MeSH terms
        logger.info('Extracting MeSH terms from the questions...')
        mm = MetamapExt()
        for q in tqdm(questions, desc='MetaMap runs'):
            rec = mm.get_mesh_descriptors(q['body'])
            tags = set()
            for concept in rec:
                entry = {
                    'source': "MetaMap",
                    'cid': concept[1],
                    'name0': concept[2].replace(';', ''),
                    'nameN': concept[2].replace(';', '')
                }
                if q['id'] not in self.concepts:
                    self.concepts[q['id']] = []
                if concept[1] not in tags:
                    tags.add(concept[1])
                    self.concepts[q['id']].append(entry)

    def run_go_expand(self):
        """This does not require the list of questions.
        Instead, it reads only the gene concepts that already retrieved from
        the TmTools concepts, and obtain GO concepts via golr service. The
        results are used to expand the existing concepts with respect to gene
        ontology."""
        logger.info('Expanding gene ontology concepts...')
        genes = dict()
        for qid, concepts in self.concepts.items():
            for c in concepts:
                if c['source'] in ['TmTools (Gene)', 'TmTools (Mutation)']:
                    for ptn in ['Species']:
                        if ptn in c['cid']:
                            continue
                    genes[c['name0']] = qid
        if len(genes) <= 0:
            return
        # Prepare GO request
        golr_url = "http://golr.geneontology.org/select?"
        params = {
            'defType': 'edismax',  # use solr's extended DisMax Query Parser
            'qt': 'standard',      # query type
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
            # Query Fields, specifies the fields in the index on which
            # to perform the query
            'qf': ['annotation_class^3',
                   'annotation_class_label_searchable^5.5',
                   'description_searchable^1',
                   'synonym_searchable^1',
                   'alternate_id^1']
        }
        # Request
        for g in tqdm(genes.keys(), desc="golr requests"):
            params['q'] = g
            for k, v in params.items():
                if type(v) is list:
                    for entry in v:
                        e = urllib.parse.quote_plus(str(entry))
                        golr_url += "{}={}&".format(k, e)
                else:
                    e = urllib.parse.quote_plus(str(v))
                    golr_url += "{}={}&".format(k, e)
            r = requests.get(golr_url)
            time.sleep(10)
            if r.status_code == 200:
                rst = json.loads(r.text)
            else:
                rst = None
                logger.error("GO API request failed: %s" % r.text)

            if rst:
                # Process the response
                num_found = rst['response']['numFound']
                if num_found == 0:
                    continue
                docs = rst['response']['docs']
                cnt = 0
                for d in docs:
                    print(g, genes[g], d['id'])
                    cnt += 1
                    entry = {
                        'source': "GO",
                        'cid': d['id'],
                        'name0': g,
                        'nameN': d['annotation_class_label']
                    }
                    self.concepts[genes[g]].append(entry)
                    if cnt > 10:
                        break

    def update_database(self):
        records = []
        qids = []
        for k, v in tqdm(self.concepts.items(), desc="Concepts DB"):
            if k not in qids:
                qids.append(k)
            for c in v:
                records.append((k, c['source'], c['cid'], c['name0'],
                                c['nameN']))
        # Delete existing rows with the question ids
        cnx = sqlite3.connect(self.db)
        # csr = cnx.cursor()
        # csr.execute("DELETE FROM concepts2 WHERE id in ({})"
        #             "".format(', '.join(['?' for _ in qids])), qids)
        # logger.info('{} concepts are deleted.'.format(csr.rowcount))
        # cnx.commit()
        if len(records) > 0:
            logger.info('Storing concepts into db...')
            csr = cnx.cursor()
            csr.executemany("INSERT OR REPLACE INTO concepts2 "
                            "VALUES (?,?,?,?,?);", records)
            logger.info('{} concepts stored in DB'.format(csr.rowcount))
            cnx.commit()
        cnx.close()

    def read_from_db(self, questions):
        assert len(self.concepts) == 0
        records = [q['id'] for q in questions]
        cnx = sqlite3.connect(self.db)
        csr = cnx.cursor()
        csr.execute("SELECT * FROM concepts2 WHERE id in ({})"
                    "".format(', '.join(['?' for _ in records])), records)
        concepts = csr.fetchall()
        for c in concepts:
            (qid, source, cid, name0, nameN) = c
            entry = {
                'source': source, 'cid': cid, 'name0': name0, 'nameN': nameN
            }
            if qid not in self.concepts:
                self.concepts[qid] = []
            self.concepts[qid].append(entry)
        return self.concepts

    def get_concepts(self, questions):
        logger.info("Retrieving concepts from the questions...")
        # questions = questions[:10]
        # If not updating the database, look for the concepts in local database
        # entries. Otherwise, use data API services.
        if not self.updateDatabase:
            return self.read_from_db(questions)

        # TmTools
        self.run_tmtools(questions)
        # MetaMap
        self.run_metamap(questions)
        # GO (Gene Ontology)
        self.run_go_expand()

        # Write to the concepts database
        if self.updateDatabase:
            self.update_database()

        return self.concepts
