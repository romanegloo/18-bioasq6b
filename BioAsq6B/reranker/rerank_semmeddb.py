"""Return a SemMedDB reranking scores of a document from the given list of
documents with respect to the query

The Semantic MEDLINE Database (SemMedDB) s a repository of semantic predications
(subject-predicate-object triples) extracted by SemRep, a semantic interpreter
of biomedical text.
This class generates a feature by utilizing SemMedDB which indicates whether
a question is relevant to a PubMed document. The assumption is that if a
sentence has the concepts shown in a question as the subject and objects,
then the sentence has a higher chance of answering the question.
Two approaches can exist; 1) By the given concepts of a questions, search all
the predication entries to find a set of documents. In this case, we may be
able to find documents that were not found by galago search, but these
documents will not have retrieval scores. 2) By the given set of relevant
documents retrieved by Galago, we can score the documents by examining
whether it contains a sentence that has both the relevant subject and object.
Another score can be obtained by examining whether the document contains all of
the concepts in it.
"""

import logging
import pymysql.cursors
from itertools import combinations
from collections import Counter

from .. import PATHS
from ..cache import Cache

logger = logging.getLogger()


class RerankerSemMedDB(object):
    def __init__(self, cache=None):
        self.cnx = None
        self.cache = cache
        self.db_connect()

    def db_connect(self):
        if self.cnx is not None and self.cnx.open:
            return
        # Read DB credential
        try:
            with open(PATHS['mysqldb_cred_file']) as f:
                host, user, passwd, dbname = f.readline().split(',')
            self.cnx = pymysql.connect(
                host=host, user=user, password=passwd, db=dbname,
                charset='utf8', cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=60*60
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
                logger.debug('SemMedDB connected')

    def db_close(self):
        if self.cnx and self.cnx.open:
            self.cnx.close()

    def get_cuis(self, qid):
        self.db_connect()
        with self.cnx.cursor() as cursor:
            # Get the list of non-generic concepts from the concepts DB
            sql = (
                "SELECT * FROM BIOASQ_CONCEPT AS cpt"
                "  LEFT OUTER JOIN GENERIC_CONCEPT AS gen"
                "  ON (cpt.cid = gen.CUI)"
                "  WHERE cpt.id=%s AND cpt.source=%s AND gen.CUI IS NULL"
            )
            cursor.execute(sql, (qid, 'MetaMap (CUI)'))
            res = cursor.fetchall()
        return [item['cid'] for item in res]

    def get_candidate_docs(self, qid=None, cuis=None):
        """Given a list of CUI concepts, find a set of documents that
        contains the possible predications of the concepts"""
        if qid is not None:
            cuis = self.get_cuis(qid)
        if len(cuis) < 2:
            return []
        docs_count = Counter()
        self.db_connect()
        with self.cnx.cursor() as cursor:
            for pair in combinations(cuis, 2):
                sql = (
                    "SELECT DISTINCT PMID FROM PREDICATION"
                    "  WHERE (SUBJECT_CUI LIKE %s AND OBJECT_CUI LIKE %s)"
                    "    OR (SUBJECT_CUI LIKE %s AND OBJECT_CUI LIKE %s)"
                )
                param1 = '{}%'.format(pair[0])
                param2 = '{}%'.format(pair[1])
                cursor.execute(sql, (param1, param2, param2, param1))
                res = cursor.fetchall()
                pmids = [item['PMID'] for item in res]
                docs_count.update(pmids)
        if sum(docs_count.values()) == 0:
            return []
        highest_cnt = docs_count.most_common(1)[0][1]
        len_first_group = sum(i == highest_cnt for i in docs_count.values())
        if len_first_group > 30:
            return []
        else:
            return docs_count.most_common(30)

    def get_predications(self, qid=None, docs=None):
        """Given either a question id or a list of documents, return SemMedDB
        features and its score; 1) if any of the CUI pairs of a question
        matches the document predication entries, 2) how many of the question
        CUIs are found in the document"""
        # If qid is given, read ranked list of documents from the cached scores
        if qid is not None:
            # check if cache scores exists. If not read from the file
            if self.cache is None:
                self.cache = Cache()
            docs = self.cache.scores[qid]['rankings']
        # Otherwise, list of docs must be given to compute the score
        if len(docs) <= 0:
            return

        # Get the list of query concepts (CUIs)
        cuis = self.get_cuis(qid)
        if len(cuis) == 0:
            return 0, 0
        self.db_connect()
        scores = []
        for doc in docs:
            # feature 1
            with self.cnx.cursor() as cursor:
                sql = "SELECT * FROM PREDICATION WHERE PMID=%s AND " \
                      "  SUBJECT_CUI RLIKE %s AND OBJECT_CUI RLIKE %s limit 1;"
                cui_pattern = '|'.join(cuis)
                cursor.execute(sql, (doc, cui_pattern, cui_pattern))
                ft_1 = 1 if cursor.rowcount >= 1 else 0

            # feature 2
            with self.cnx.cursor() as cursor:
                sql = """SELECT COUNT(uniontable.cui) AS cnt FROM (
                    SELECT SUBJECT_CUI as cui FROM PREDICATION WHERE PMID=%s
                    UNION 
                    SELECT OBJECT_CUI as cui FROM PREDICATION WHERE PMID=%s
                    ) uniontable WHERE uniontable.cui RLIKE %s;"""
                cursor.execute(sql, (doc, doc, '|'.join(cuis)))
                ft_2 = int(cursor.fetchone()['cnt']) / len(cuis)
                scores.append((ft_1, ft_2))
        return scores

    def get_semmeddb_scores(self, ranked_docs, cache=None):
        """From the question and the ranked list of documents, return
        SemMedDB feature scores"""

        # If cached scores exist, return them
        k = len(ranked_docs.rankings)
        if cache is not None and 'scores-semmeddb' in cache \
                and cache['scores-semmeddb'] is not None:
            if len(cache['scores-semmeddb']) >= k:
                ranked_docs.scores['semmeddb'] = cache['scores-semmeddb'][:k]
                return
        # Get the list of query concepts (CUIs)
        q = ranked_docs.query
        cuis = self.get_cuis(q['id'])
        if len(cuis) == 0:
            return
        self.db_connect()
        scores = [(0, 0)] * k
        for i, doc in enumerate(ranked_docs.rankings):
            # feature 1
            with self.cnx.cursor() as cursor:
                sql = "SELECT * FROM PREDICATION WHERE PMID=%s AND " \
                      "  SUBJECT_CUI RLIKE %s AND OBJECT_CUI RLIKE %s limit 1;"
                cui_pattern = '|'.join(cuis)
                cursor.execute(sql, (doc, cui_pattern, cui_pattern))
                ft_1 = 1 if cursor.rowcount >= 1 else 0

            # feature 2
            with self.cnx.cursor() as cursor:
                sql = """SELECT COUNT(uniontable.cui) AS cnt FROM (
                    SELECT SUBJECT_CUI as cui FROM PREDICATION WHERE PMID=%s
                    UNION 
                    SELECT OBJECT_CUI as cui FROM PREDICATION WHERE PMID=%s
                    ) uniontable WHERE uniontable.cui RLIKE %s;"""
                cursor.execute(sql, (doc, doc, '|'.join(cuis)))
                ft_2 = int(cursor.fetchone()['cnt']) / len(cuis)
            scores[i] = (ft_1, ft_2)
        ranked_docs.scores['semmeddb'] = scores
        ranked_docs.update_cache.append('semmeddb')
