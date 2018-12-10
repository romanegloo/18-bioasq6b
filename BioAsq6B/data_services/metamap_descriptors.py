"""run metamap and extract descriptors from given queries

note. make sure that the part-of-speech tagger server and word send
disambiguation server is started.
./bin/skrmedpostctl start
./bin/wsdserverctl start
"""

from subprocess import run, PIPE, STDOUT
import logging
import re
import pymysql

from BioAsq6B import PATHS

logger = logging.getLogger()


class MetamapExt(object):
    FIELD_NAME_MMI = ('id', 'mmi', 'score', 'preferred_name', 'cui',
                      'sem_type', 'trigger_info', 'location',
                      'positional_info', 'treecode')

    def __init__(self):
        self.mm = PATHS['metamap_bin']
        self.mm_command = [
            self.mm,
            '-N',           # Fielded MMI Output
            '-y',           # word sense disambiguation
            '--conj',       # turn on conjunction processing
            '--silent',     # suppress the display of header information
            '--cascade'     # cascade concept deletion
        ]
        # Read DB credential
        with open(PATHS['mysqldb_cred_file']) as f:
            host, user, passwd, dbname = f.readline().split(',')
        try:
            self.cnx = pymysql.connect(
                host=host, user=user, password=passwd, db=dbname,
                charset='utf8', cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.cnx.cursor()
        except Exception as e:
            logger.error('DB connection failed: {}'.format(e))
            raise

    def get_mesh_names(self, query):
        """Run metamap and get UMLS preferred names and mesh heading names for
        building expanded query for document retrieval"""
        # query += "\n" if not query.endswith("\n") else ''
        preferred_names = []
        mesh = []
        p = run(self.mm_command, stdout=PIPE, stderr=STDOUT, input=query,
                universal_newlines=True)
        if p.returncode != 0:
            logger.warning("failed to run metamap: returned {}"
                           "".format(p.returncode))
        else:
            # parse the returned message and return a list of mesh records
            for line in p.stdout.splitlines():
                fields = line.split('|')
                if len(fields) == len(MetamapExt.FIELD_NAME_MMI) and \
                        fields[1] == 'MMI':
                    fields_named = dict(zip(MetamapExt.FIELD_NAME_MMI, fields))
                    pn = fields_named['preferred_name']
                    pn = re.sub(r"[^\w\s]", '', pn)
                    preferred_names.append(pn)
                    codes = fields_named['treecode'].split(';')
                    if len(codes) > 0:
                        mh = self.get_meshHeadings(codes)
                        for rec in mh:
                            mesh.append(re.sub(r"[^\w\s]", '', rec[2]))
                else:
                    continue
        return list(set(preferred_names) | set(mesh))

    def get_concepts(self, query):
        """Run MetaMap and return the list of CUIs and MeSH descriptors"""
        query += "\n" if not query.endswith("\n") else ''
        cuis = []
        meshes = []

        p = run(self.mm_command, stdout=PIPE, stderr=STDOUT, input=query,
                universal_newlines=True)
        if p.returncode != 0:
            logger.warning('MetaMap Runtime Error {}'.format(p.returncode))
        else:
            # parse the returned message and return a list of mesh records
            for line in p.stdout.splitlines():
                fields = line.split('|')
                if len(fields) == len(MetamapExt.FIELD_NAME_MMI):
                    fields_named = dict(zip(MetamapExt.FIELD_NAME_MMI, fields))
                    cuis.append((fields_named['cui'],
                                 re.sub(r"[^\w\s]", '',
                                        fields_named['preferred_name'])))
                    treecodes = fields_named['treecode'].split(';')
                    if len(treecodes) > 0:
                        mh = self.get_meshHeadings(treecodes)
                        meshes.extend(mh)
                else:
                    continue
        return cuis, meshes

    def get_meshHeadings(self, codes):
        """
        from the obtained tree code, get the name of mesh descriptor and id.
        Mapping database is built by running scripts/documents/build_mesh_db.py
        :param codes: list of tree code (e.g. ['C10.228.140.490', ...])
        :return: list of MESH name
                (e.g. [('C10.228.140.490', 'D123456', 'Epileps'), ...])
        """
        meshHeadings = []
        SQL = ("SELECT t.*, d.name FROM MESH_TREECODE AS t "
               "INNER JOIN MESH_NAME AS d ON t.mesh_id = d.mesh_id "
               "WHERE t.tui = '%s';")
        for code in codes:
            if len(code.strip()) <= 0:
                continue
            logger.debug("looking up meshHeading for %s" % code)
            self.cursor.execute(SQL % code)
            for rec in self.cursor.fetchall():
                meshHeadings.append(rec)
        return meshHeadings
