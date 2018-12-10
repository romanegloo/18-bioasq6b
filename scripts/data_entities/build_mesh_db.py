#!/usr/bin/env python3
"""A script to read mesh descriptor datafile in XML and store records in a
sqlite database.

descriptor: [DescriptorUI, DescriptorName]
treecode: [DescriptorUI, TreeNumber]
"""
import os
import pymysql
from BioAsq6B import PATHS

def store_contents(data_file, db_file):
    """[!] USED ONCE, NEEDS TO REWRITE WHEN TO RUN
    Read MESH descriptor file ( supplementary, qualifiers are not
    allowed), and store descriptors in sqlite"""

    raise NotImplemented
    # if os.path.isfile(db_file):
    #     raise RuntimeError("%s already exists! Not overwriting." % db_file)
    # if not os.path.isfile(data_file):
    #     raise RuntimeError("data file does not exist: %s" % data_file)
    #
    # print('Reading data file...')
    # cnx = sqlite3.connect(db_file)
    # csr = cnx.cursor()
    # csr.execute("CREATE TABLE descriptor (dui text PRIMARY KEY, name text);")
    # csr.execute("CREATE TABLE treecode (tui text, dui text, "
    #             "PRIMARY KEY (tui, dui));")
    #
    # # read descriptor file
    # try:
    #     dataset = etree.parse(data_file)
    # except IOError:
    #     print("lxml: cannoter read data file")
    #     return
    # except etree.XMLSyntaxError:
    #     print("lxml: cannot parse, syntax error")
    #     return
    #
    # descriptors = {}
    # treecodes = {}
    # for d in dataset.getiterator('DescriptorRecord'):
    #     dui = d.find("DescriptorUI").text
    #     descriptors[dui] = d.find("DescriptorName/String").text
    #
    #     for code in d.findall('.//TreeNumber'):
    #         treecodes[code.text] = dui
    #
    # print("Inserting descriptors...")
    # csr.executemany("INSERT INTO descriptor VALUES (?,?)", descriptors.items())
    # print("Read %d descriptors." % len(descriptors))
    # print("Commiting...")
    # cnx.commit()
    #
    # print("Inserting tree codes...")
    # csr.executemany("INSERT INTO treecode VALUES (?,?)", treecodes.items())
    # print("Read %d tree codes." % len(treecodes))
    # print("Commiting...")
    # cnx.commit()
    #
    # cnx.close()

def insert_mti_mesh():
    """Read mti results of all the questions, insert them into db
    data format:
    qid|PRC|docid1;docid2;...;docidn
    qid|mesh name|mesh id|cui|score
    """
    data_file = os.path.join(PATHS['data_dir'], 'all_questions-mti_mod.out')
    # Connect db
    try:
        with open(PATHS['mysqldb_cred_file']) as f:
            host, user, passwd = f.readline().strip().split(',')
            cnx = pymysql.connect(
                host=host, user=user, password=passwd, db='jno236_ir',
                charset='utf8', cursorclass=pymysql.cursors.DictCursor,
            )
    except (pymysql.err.DatabaseError,
            pymysql.err.IntegrityError,
            pymysql.err.MySQLError) as exception:
        raise RuntimeError('DB connection failed: {}'.format(exception))
    records = []
    with open(data_file) as f:
        for line in f:
            fields = line.strip().split('|')
            if fields[1] != 'PRC' and int(fields[4]) > 300:
                records.append((fields[0], 'MTI (MeSH)', fields[2], fields[1],
                                fields[1]))
    assert len(records) > 0
    with cnx.cursor() as csr:
        sql = "INSERT INTO BIOASQ_CONCEPT VALUES (%s, %s, %s, %s, %s);"
        print("Inserting records into table [BIOASQ_CONCEPT]...")
        try:
            csr.executemany(sql, records)
        except pymysql.err.MySQLError as e:
            print('DB error: {}'.format(e))
        else:
            cnx.commit()
            print("Completed. updated rows ({})".format(csr.rowcount))
    cnx.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # store_contents(args.data_file, args.db_file)
    insert_mti_mesh()

