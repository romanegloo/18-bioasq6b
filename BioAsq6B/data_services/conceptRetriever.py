"""Full pipeline of getting a list of relevant concepts from the given
question text"""

import logging
import json
import requests
import urllib.parse
import sys
from tqdm import tqdm
import pymysql
import time

from . import MetamapExt
from .. import PATHS

# initialize logger
logger = logging.getLogger()
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ConceptRetriever(object):
    def __init__(self, updateDatabase=False):
        self.concepts = list()
        self.updateDatabase = updateDatabase
        self.cnx = None
        self.db_connect()

    def db_connect(self):
        # Read DB credential
        if self.cnx is not None and self.cnx.open:
            return
        # Read DB credential
        try:
            with open(PATHS['mysqldb_cred_file']) as f:
                host, user, passwd, dbname = f.readline().split(',')
            self.cnx = pymysql.connect(
                host=host, user=user, password=passwd, db=dbname,
                charset='utf8', cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=2*60*60
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
                logger.debug('Concept DB connected')

    def db_close(self):
        if self.cnx and self.cnx.open:
            self.cnx.close()

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
            qids = [c['id'] for c in self.concepts]
            for q_resp in data:
                for concept in q_resp['denotations']:
                    b = concept['span']['begin']
                    e = concept['span']['end']
                    term = q_resp['text'][b:e]
                    # term = term.replace(';', ' ')
                    entry = {
                        'id': q_resp['sourceid'],
                        'source': "TmTools ({})".format(trigger),
                        'cid': concept['obj'],
                        'name_0': term,
                        'name_norm': term
                    }
                    self.concepts.append(entry)

    def run_metamap(self, questions):
        """Runs MataMap to extract MeSH concepts"""
        # get MeSH terms
        logger.info('Extracting MeSH terms from the questions...')
        mm = MetamapExt()
        for q in tqdm(questions, desc='MetaMap Runs'):
            cuis, meshes = mm.get_concepts(q['body'])
            # CUIs
            for concept in cuis:
                entry = {
                    'id': q['id'],
                    'source': 'MetaMap (CUI)',
                    'cid': concept[0],
                    'name_0': concept[1].replace(';', ''),
                    'name_norm': concept[1].replace(';', '')
                }
                self.concepts.append(entry)
            # MeSHes
            for concept in meshes:
                tags = set()
                entry = {
                    'id': q['id'],
                    'source': 'MetaMap (MeSH)',
                    'cid': concept['mesh_id'],
                    'name_0': concept['name'].replace(';', ''),
                    'name_norm': concept['name'].replace(';', '')
                }
                if concept['mesh_id'] not in tags:
                    tags.add(concept['mesh_id'])
                    self.concepts.append(entry)

    def run_go_expand(self):
        """This does not require the list of questions.
        Instead, it reads only the gene concepts that already retrieved from
        the TmTools concepts, and obtain GO concepts via golr service. The
        results are used to expand the existing concepts with respect to gene
        ontology."""
        logger.info('Expanding gene ontology concepts...')
        genes = dict()
        for c in self.concepts:
            if c['source'] in ['TmTools (Gene)', 'TmTools (Mutation)']:
                for ptn in ['Species']:
                    if ptn in c['cid']:
                        continue
                genes[c['name_0']] = c['id']
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

            if r:
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
                        'id': genes[g],
                        'source': "GO",
                        'cid': d['id'],
                        'name_0': g,
                        'name_norm': d['annotation_class_label']
                    }
                    self.concepts.append(entry)
                    if cnt > 10:
                        break

    def update_database(self):
        records = []
        for c in tqdm(self.concepts, desc="Concepts DB"):
            records.append((c['id'], c['source'], c['cid'], c['name_0'],
                            c['name_norm']))
        self.cnx = None
        self.db_connect()

        if len(records) > 0:
            with self.cnx.cursor() as cursor:
                sql = "REPLACE INTO BIOASQ_CONCEPT VALUES (%s, %s, %s, %s, %s);"
                cursor.executemany(sql, records)
                logger.info('{} concepts updated in concepts DB'
                            ''.format(cursor.rowcount))
            self.cnx.commit()
        else:
            logger.warning('\nNothing to update in concepts DB')

    def read_from_db(self, questions):
        self.db_connect()
        assert len(self.concepts) == 0
        records = [q['id'] for q in questions]
        with self.cnx.cursor() as cursor:
            sql = "SELECT * FROM BIOASQ_CONCEPT WHERE id IN %s;"
            cursor.execute(sql, (records,))
            self.concepts = list(cursor.fetchall())
        return self.concepts

    def get_concepts(self, questions):
        logger.info("Retrieving concepts from the questions...")
        # If not updating the database, look for the concepts in local database
        # entries. Otherwise, use data API services.
        self.concepts = list()
        if self.updateDatabase:
            # TmTools
            self.run_tmtools(questions)
            # MetaMap
            self.run_metamap(questions)
            # GO (Gene Ontology)
            self.run_go_expand()
            self.update_database()

        if len(self.concepts) > 0:
            return self.concepts
        else:
            return self.read_from_db(questions)
