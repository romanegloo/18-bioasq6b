import os
from .. import PATHS

DEFAULTS = {
    'mesh_desc_db': os.path.join(PATHS['data_dir'], 'mesh/desc2017.db'),
    'metamap_file': '/home/jno236/opt/public_mm/bin/metamap'
}

def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value

from .metamap_descriptors import MetamapExt
from .amigo_ontology import GO_Ext
from .bioasq_api import BioasqAPI
from .conceptRetriever import ConceptRetriever