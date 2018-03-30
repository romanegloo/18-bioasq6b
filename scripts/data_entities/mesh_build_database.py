"""Build a MeSH database from xml files; populate a mesh concepts table and a
treecode mapping table

mesh: [concept id, name, type]
treecode: [cui, tree number]
"""

from pathlib import Path
from lxml import etree
import sqlite3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

data_dir = Path(__file__).absolute().parents[3] / 'data'
db_file = data_dir / 'concepts.db'
desc_file = data_dir / 'mesh/desc2018.xml'
supp_file = data_dir / 'mesh/supp2018.xml'

logger.info("Creating tables if not exist...")
cnx = sqlite3.connect(db_file.as_posix())
csr = cnx.cursor()
csr.execute("CREATE TABLE IF NOT EXISTS mesh (cui text PRIMARY KEY, "
            "type text, name text);")
csr.execute("CREATE TABLE IF NOT EXISTS treecode (tui text, cui text, "
            "PRIMARY KEY (tui, cui));")

logger.info("Reading MeSH descriptor xml file...")
data = etree.parse(desc_file.as_posix())
concepts = {}
treecodes = {}
for rec in data.getiterator("DescriptorRecord"):
    cui = rec.find("DescriptorUI").text
    name = rec.find("DescriptorName/String").text
    concepts[cui] = name
    for treecode in rec.findall('.//TreeNumber'):
        if treecode not in treecodes:
            treecodes[treecode.text] = cui
        else:
            treecodes[treecode.text] += '/' + cui

logger.info("Inserting descriptors...")
csr.executemany("INSERT OR IGNORE INTO mesh VALUES (?,'desc', ?)",
                concepts.items())
logger.info("Read %d descriptors." % len(concepts))
logger.info("Commiting...")
cnx.commit()

logger.info("Inserting tree codes...")
csr.executemany("INSERT OR IGNORE INTO treecode VALUES (?,?)",
                treecodes.items())
logger.info("Read %d tree codes." % len(treecodes))
logger.info("Commiting...")
cnx.commit()

logger.info("Reading MeSH supplementary xml file...")
data = etree.parse(supp_file.as_posix())
concepts = {}
for rec in data.getiterator("SupplementalRecord"):
    cui = rec.find("SupplementalRecordUI").text
    name = rec.find("SupplementalRecordName/String").text
    concepts[cui] = name

logger.info("Inserting supplementals...")
csr.executemany("INSERT OR IGNORE INTO mesh VALUES (?,'supp', ?)",
                concepts.items())
logger.info("Read %d descriptors." % len(concepts))
logger.info("Commiting...")
cnx.commit()
