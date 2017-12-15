"""run metamap and extract descriptors from given queries

note. make sure that the part-of-speech tagger server and word send
disambiguation server is started.
./bin/skrmedpostctl start
./bin/wsdserverctl start
"""
#todo. programmatically start the above servers, if not running

from subprocess import run, PIPE, STDOUT
import logging
import sqlite3
import re
from . import DEFAULTS, set_default

logger = logging.getLogger()

class MetamapExt(object):
    FIELD_NAME_MMI = ('id', 'mmi', 'score', 'preferred_name', 'cui',
                      'sem_type', 'trigger_info', 'location',
                      'positional_info', 'treecode')
    FIELD_NAME_AA = ('id', 'aa', 'short_form', 'long_form',
                     'no_tokens_short', 'no_chars_short', 'no_tokens_long',
                     'no_chars_long', 'positional_info')

    def __init__(self, metamap=DEFAULTS['metamap_file'], options=[]):
        self.mm = metamap
        self.options = options

    def get_mesh_names(self, query):
        """Run metamap and get UMLS preferred names and mesh heading names for
        building expanded query for document retrieval"""
        query += "\n" if not query.endswith("\n") else ''
        preferred_names = []
        mesh = []
        command = [self.mm,
                   '-N',  #
                   '-y',  # word sense disambiguation
                   '--silent',  # suppress the display of header information
                   '--cascade',
                   ]
        p = run(command, stdout=PIPE, stderr=STDOUT, input=query,
                universal_newlines=True)
        if p.returncode != 0:
            logger.warning("failed to run metamap: returned {}"
                           "".format(p.returncode))
        else:
            # parse the returned message and return a list of mesh records
            lines = p.stdout.splitlines()
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
                elif len(fields) == len(MetamapExt.FIELD_NAME_AA) and \
                        fields[1] == 'AA':
                    fields_named = dict(zip(MetamapExt.FIELD_NAME_AA, fields))
                    preferred_names.append(fields_named['long_form'])
                else:
                    continue
        return list(set(preferred_names) | set(mesh))

    def get_mesh_descriptors(self, query):
        """Mainly used for concept retrieval. Run metamap and get the list of
        descriptors with its code and name"""
        query += "\n" if not query.endswith("\n") else ''
        mesh = []
        command = [self.mm,
                   '-N',  #
                   '-y',  # word sense disambiguation
                   '--silent',  # suppress the display of header information
                   '--cascade',
                   ]
        p = run(command, stdout=PIPE, stderr=STDOUT, input=query,
                universal_newlines=True)
        if p.returncode != 0:
            logger.warning("failed to run metamap: returned {}"
                           "".format(p.returncode))
        else:
            # parse the returned message and return a list of mesh records
            lines = p.stdout.splitlines()
            for line in p.stdout.splitlines():
                fields = line.split('|')
                if len(fields) == len(MetamapExt.FIELD_NAME_MMI) and \
                        fields[1] == 'MMI':
                    fields_named = dict(zip(MetamapExt.FIELD_NAME_MMI, fields))
                    pn = fields_named['preferred_name']
                    pn = re.sub(r"[^\w\s]", '', pn)
                    codes = fields_named['treecode'].split(';')
                    if len(codes) > 0:
                        mh = self.get_meshHeadings(codes)
                        mesh.extend(mh)
                else:
                    continue
        return mesh

    def get_meshHeadings(self, codes):
        """
        from the obtained tree code, get the name of mesh descriptor and id.
        Mapping database is built by running scripts/documents/build_mesh_db.py
        :param codes: list of tree code (e.g. ['C10.228.140.490', ...])
        :return: list of MESH name
                (e.g. [('C10.228.140.490', 'D123456', 'Epileps'), ...])
        """
        meshHeadings = []
        cnx = sqlite3.connect(DEFAULTS['mesh_desc_db'])
        csr = cnx.cursor()
        SQL = """SELECT t.*, d.name FROM `treecode` AS t INNER JOIN `descriptor`
              AS d ON t.dui = d.dui WHERE t.tui = '%s';"""
        for code in codes:
            if len(code.strip()) <= 0:
                continue
            logger.debug("looking up meshHeading for %s" % code)
            csr.execute(SQL % code)
            for rec in csr.fetchall():
                meshHeadings.append(rec)
        return meshHeadings

