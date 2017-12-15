"""BioasqAPI, which requests data to BioASQ data services"""

import requests
import os
import json
import logging

logger = logging.getLogger(__name__)
ENDPOINTS = {
    'DO':               "http://bioasq.org:8000/do",
    'GO':               "http://bioasq.org:8000/go",
    'JOCHEM':           "http://bioasq.org:8000/jochem",
    'MESH':             "http://bioasq.org:8000/mesh",
    'UNIPROT':          "http://bioasq.org:8000/uniprot",
    'LINKEDLIFEDATA':   "http://bioasq.org:8000/triples",
    'PUBMED':           "http://bioasq.org:8000"
}


class BioasqAPI(object):
    reopen_session_trial = 3

    def __init__(self):
        self.session_code = None
        self.get_session_code()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_session_code(self):
        if self.session_code is not None:
            return

        r = requests.get(ENDPOINTS['GO'])
        if r.status_code == 200:
            print("API session created: %s" % r.text)
            self.session_code = r.text.split('/')[-1]
        else:
            print("[API Error]", r.text)

    def get_concepts(self, keywords, page=1, per_page=10, source_type='MESH'):
        """the results of mesh concepts are not promising: may need to find
        an alternative way"""
        req = {'findEntitiesPaged': [keywords, page, per_page]}
        url = os.path.join(ENDPOINTS[source_type], self.session_code)
        return self._post(url, req)

    def get_articles(self, keywords, page=1, per_page=3):
        req = {'findPubMedCitations': [keywords, page, per_page]}
        url = os.path.join(ENDPOINTS['PUBMED'], self.session_code)
        return self._post(url, req)

    def get_RDFs(self, keywords, page=1, per_page=3):
        req = {'findEntitiesPaged': [keywords, page, per_page]}
        url = os.path.join(ENDPOINTS['LINKEDLIFEDATA'], self.session_code)
        return self._post(url, req)

    def _post(self, url, payload):
        r = requests.post(url, data={'json': json.dumps(payload)})
        rst = json.loads(r.text)
        if 'exception' in rst:
            print("[API Error] %s" % rst['exception']['description'])
            print("request: %s" % payload)
        elif 'result' in rst:
            return rst
