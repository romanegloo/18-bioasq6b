"""request query to gene ontology backend data service, to obtain gene
ontology_class concepts

query for GeneOntology (GO) concepts is prepared by combining following
concept names:
- MeSH concepts (by
"""
import nltk
import urllib.parse
import requests
import json
import os
import subprocess
import sqlite3
import logging
from .. import PATHS

logger = logging.getLogger()


class GO_Ext(object):
    def __init__(self, np_pattern=None):
        if np_pattern is None:
            self.np_pattern = """
                NP:    {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
                       {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNP>+}
                       {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                       {<JJ>*<NN|NNS|NNP|NNPS>+}"""
        else:
            self.np_pattern = np_pattern

        self.chunker = nltk.RegexpParser(self.np_pattern)
        self.db_path = os.path.join(PATHS['data_dir'], 'var/go')
        self.db_file = os.path.join(self.db_path, 'go_query_cache.db')
        if not os.path.exists(self.db_path):
            logger.warning("GO query cache file does not exist: %s" %
                           self.db_path)
            self._build_cache_db()

    def _build_cache_db(self):
        try:
            subprocess.call(['mkdir', '-p', self.db_path])
        except subprocess.CalledProcessError as e:
            logger.error("Error while creating GO cache db path")
        else:
            logger.info("creating GO cache db file")
            cnx = sqlite3.connect(self.db_file)
            csr = cnx.cursor()
            csr.execute("CREATE TABLE q_resp (query text PRIMARY KEY, "
                        "response text);")
            cnx.close()

    def _chunk_parse(self, text):
        tok_sentence = nltk.sent_tokenize(text)
        tok_words = [nltk.word_tokenize(s) for s in tok_sentence]
        tagged_words = [nltk.pos_tag(w) for w in tok_words]
        word_tree = [self.chunker.parse(w) for w in tagged_words]
        return word_tree

    def extract_noun_phrase(self, sentences):
        parsed_sentences = self._chunk_parse(sentences)
        nps = []
        for s in parsed_sentences:
            tree = self.chunker.parse(s)
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    t = subtree
                    t = ' '.join(word for word, tag in t.leaves())
                    nps.append(t)
        return nps

    def get_go_concepts(self, question_str):
        # get the noun phrase
        # todo. may need to preprocess the phrase, such as (solr's AND,
        # OR operators)
        for q_phrase in self.extract_noun_phrase(question_str):
            # check if the phrase is queried and the result is cached
            cnx = sqlite3.connect(self.db_file)
            csr = cnx.cursor()
            print(q_phrase)
            csr.execute("SELECT * FROM q_resp where query=?",
                        (q_phrase,))
            result = csr.fetchone()
            if result is None:
                result = self._request(q_phrase)
            self._extract_concept(result)

    def _request(self, q):
        golr_url = "http://golr.geneontology.org/select?"
        params = {
            'defType': 'edismax',  # use solr's extended DisMax Query Parser
            'qt': 'standard',      # query type
            # 'indent': 'ON',        # indentation of the response
            'wt': 'json',          # specifies the response format, default: XML
            'rows': '10',            # maximum number of returned records
            'start': '0',            # paging
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
            'q': q + '*',
            # Query Fields, specifies the fields in the index on which
            # to perform the query
            'qf': ['annotation_class^3',
                   'annotation_class_label_searchable^5.5',
                   'description_searchable^1',
                   'synonym_searchable^1',
                   'alternate_id^1']
        }
        url = golr_url
        for k, v in params.items():
            if type(v) is list:
                for entry in v:
                    e = urllib.parse.quote_plus(entry)
                    url += "{}={}&".format(k, e)
            else:
                e = urllib.parse.quote_plus(v)
                url+= "{}={}&".format(k, e)

        r = requests.get(url)
        if r.status_code == 200:
            rst = json.loads(r.text)
            print(rst)
        else:
            logger.error("GO api request failed: %s" % r.text)

    def _extract_concept(self, result_str):
        print(result_str)