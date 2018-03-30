#!/usr/bin/env python3

"""A script to read mesh descriptor datafile in XML and store records in a
sqlite database.

descriptor: [DescriptorUI, DescriptorName]
treecode: [DescriptorUI, TreeNumber]
"""

import argparse
import sqlite3
from lxml import etree
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def store_contents(data_file, db_file):
    """Read MESH descriptor file (supplementary, qualifiers are not allowed),
    and store descriptors in sqlite"""

    if os.path.isfile(db_file):
        raise RuntimeError("%s already exists! Not overwriting." % db_file)
    if not os.path.isfile(data_file):
        raise RuntimeError("data file does not exist: %s" % data_file)

    logger.info('Reading data file...')
    cnx = sqlite3.connect(db_file)
    csr = cnx.cursor()
    csr.execute("CREATE TABLE descriptor (dui text PRIMARY KEY, name text);")
    csr.execute("CREATE TABLE treecode (tui text, dui text, "
                "PRIMARY KEY (tui, dui));")

    # read descriptor file
    try:
        dataset = etree.parse(data_file)
    except IOError:
        logger.error("lxml: cannoter read data file")
        return
    except etree.XMLSyntaxError:
        logger.error("lxml: cannot parse, syntax error")
        return

    descriptors = {}
    treecodes = {}
    for d in dataset.getiterator('DescriptorRecord'):
        dui = d.find("DescriptorUI").text
        descriptors[dui] = d.find("DescriptorName/String").text

        for code in d.findall('.//TreeNumber'):
            treecodes[code.text] = dui

    logger.info("Inserting descriptors...")
    csr.executemany("INSERT INTO descriptor VALUES (?,?)", descriptors.items())
    logger.info("Read %d descriptors." % len(descriptors))
    logger.info("Commiting...")
    cnx.commit()

    logger.info("Inserting tree codes...")
    csr.executemany("INSERT INTO treecode VALUES (?,?)", treecodes.items())
    logger.info("Read %d tree codes." % len(treecodes))
    logger.info("Commiting...")
    cnx.commit()

    cnx.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='/path/to/data_file')
    parser.add_argument('db_file', type=str, help='/path/to/db_file')
    args = parser.parse_args()

    store_contents(args.data_file, args.db_file)
