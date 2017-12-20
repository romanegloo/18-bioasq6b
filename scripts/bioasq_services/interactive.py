#!/usr/bin/env python3
"""interactively runs BioasqAPI to retrieve data fro BioASQ data services"""

import code
import argparse
import logging
import prettytable

from BioAsq6b.bioasq_services import BioasqAPI

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


API = BioasqAPI()


def process(resource, keywords, page=1, per_page=10, source_type=None):
    if resource == 'concept':
        r = API.getConcepts(keywords, page, per_page, source_type)
    elif resource == 'article':
        r = API.getArticles(keywords, page, per_page)
    elif resource == 'rdf':
        r = API.getRDFs(keywords, page, per_page)
    else:
        logger.error("unknow resource type: %s" % resource)
        return
    print(r)

banner = """
BioasqAPI retrieving data from BioASQ data services

process(resource, keywords, page=1, per_page=10, source_type=None):
    - resource in ['concept', 'article', 'rdf']
    - for concepts, 
        specify source_type from ['DO', 'GO', 'JOCHEM', 'MESH', 'UNIPROT']

e.g. process('article', "Arthritis Rheumotoid")

usage()
"""


def usage():
    print(banner)

code.interact(banner, local=locals())