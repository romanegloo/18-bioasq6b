"""Testing Cache object"""

import code
import logging

from BioAsq6B.cache import Cache

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

cache = Cache()

banner = """
usage:
  >>> cache.get_scores_retrieval('5a6d1db1b750ff4455000033', k=15)
"""
def usage():
    print(banner)

code.interact(banner, local=locals())
