"""Given a query and the list of document sources (journals), get the reranking
scores from the journal meshes with respect to the query meshes

The journal mesh distribution table has the following entities:
    - journals:
        the journal list with its descriptive information including
        '[key]:MedlineTA', 'issn:ISSN', 'title:TitleMain/Title',
        'mesh:MeshHeadingList/MeshHeading', 'language:Language'
    - qmesh2idx: mapping query meshes to indices
    - idx2qmesh: mapping indices to query meshes
    - jmesh2idx: mappingn journal meshes to indices
    - idx2jmesh: mapping indices to journal meshes
    - dist_table: distribution table between qmesh and jmesh
"""

import os
import pickle
import logging
import pymysql
from sklearn.preprocessing import normalize

from BioAsq6B import PATHS
from BioAsq6B.data_services import ConceptRetriever

logger = logging.getLogger()


class RerankerJournal(object):
    table_file = os.path.join(PATHS['data_dir'], 'journal_mesh_dist.pkl')

    def __init__(self):
        logger.info("Initializing a Journal reranker")
        logger.info("Loading journal mesh distribution table...")
        try:
            self.data = pickle.load(open(RerankerJournal.table_file, 'rb'))
        except:
            raise RuntimeError("Cannot load the journal distribution data [{}]"
                               "".format(RerankerJournal.table_file))
        self.concept_retriever = None
        self.dist = self.data['dist_table']
        self.cnx = None
        self.db_connect()

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
                logger.debug('Database for MeSH connected')

    def db_close(self):
        if self.cnx and self.cnx.open:
            self.cnx.close()
            pymysql.connect().close()

    def get_mesh_name(self, mesh):
        with self.cnx.cursor() as cursor:
            sql = "SELECT * FROM MESH_NAME WHERE mesh_id='{}';".format(mesh)
            cursor.execute(sql)
            concept = cursor.fetchone()
        if concept is not None:
            return concept['name']
        return 'N/A'

    def get_jmeshes(self, qmesh):
        if qmesh in self.data['qmesh2idx']:
            qidx = self.data['qmesh2idx'][qmesh]
            print("{} \"{}\"".format(qmesh, self.get_mesh_name(qmesh)))
            for jmesh in [(i, cnt) for i, cnt in enumerate(self.dist[qidx])
                          if cnt > 0]:
                jmesh_id = self.data['idx2jmesh'][jmesh[0]]
                print("  - {} \"{}\" ({})"
                      "".format(jmesh_id, self.get_mesh_name(jmesh_id),
                                jmesh[1]))
        else:
            print('{} not found'.format(qmesh))
            return ''

    def get_journal_scores(self, ranked_docs):
        """From the list of document, get the journal sources and its
        relevance scores with respect to the given query"""
        k = len(ranked_docs.rankings)
        # Get the list of query meshes (qmeshes)
        q = ranked_docs.query
        if self.concept_retriever is None:
            self.concept_retriever = ConceptRetriever()
        concepts = self.concept_retriever.get_concepts([q])
        if len(concepts) <= 0:
            return
        scores = [0] * k
        # Process qmeshes
        qmeshes = []
        for concept in concepts:
            if concept['source'] == 'MTI (MeSH)':
                qmeshes.append(concept['cid'])
            else:
                c_ = concept['cid'].split(':')
                if len(c_) == 3 and c_[1] == 'MESH':
                    qmeshes.append(c_[2])

        logger.info("qmeshes: {}".format(qmeshes))
        q_indices = list(set([self.data['qmesh2idx'][qmesh] for qmesh in qmeshes
                              if qmesh in self.data['qmesh2idx']]))
        logger.info("q-indices: {}".format(q_indices))
        if len(q_indices) == 0:
            return
        frequencies = normalize(self.dist[q_indices], axis=1)

        # From the ranked_docs, get the journal name and its descriptive
        # meshes (jmeshes)
        for i, docid in enumerate(ranked_docs.rankings):
            if docid not in ranked_docs.docs_data:
                print('not found')
                continue
            journal_title = ranked_docs.docs_data[docid]['journal']
            if journal_title not in self.data['journals']:
                continue
            journal_meshes = self.data['journals'][journal_title]['mesh']
            j_indices = [self.data['jmesh2idx'][jmesh]
                         for jmesh in journal_meshes]
            scores[i] = frequencies[:, j_indices].sum() / len(q_indices)
        ranked_docs.scores['journal'] = scores

