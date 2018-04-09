"""Reads a training dataset and build the QueryMeshes to JournalMeshes
distribution"""
from pathlib import Path
import json
from lxml import etree
import sqlite3
import numpy as np
import subprocess
import re
from multiprocessing import Pool, cpu_count
import pickle

data_dir = Path(__file__).absolute().parents[3] / 'data'
t_file = (data_dir / 'bioasq/train/BioASQ-trainingDataset6b.json').as_posix()
j_file = (data_dir / 'nlmcatalog_result.xml').as_posix()
db_file = (data_dir / 'concepts.db').as_posix()
galago_idx = (data_dir / 'galago-medline-full-idx').as_posix()
output_file = data_dir / 'journal_mesh_dist.pkl'
output_overwrite = False
if output_file.exists():
    confirm = input("output file exist. overwrite [Y/n]? ")
    if not confirm.lower().startswith('n'):
        output_overwrite = True
else:
    output_overwrite = True

# Build Journal list
#   key: MedlineTA, values: [ISSN@Print, Title, MeSHes, Language]
print("Parsing journal list...")
with open(j_file) as f:
    journals_xml = etree.parse(f)
journals = dict()
for rec in journals_xml.iter("NLMCatalogRecord"):
    entry = dict()
    # key (MedlineTA)
    key = rec.find("MedlineTA")
    if key is None:
        continue
    # ISSN (valid and print)
    issn = rec.find("ISSN[@ValidYN='Y'][@IssnType='Print']")
    # Title
    title = rec.find("TitleMain/Title")
    # MeSH CUIs
    meshes = []
    for m in rec.findall("MeshHeadingList/MeshHeading"):
        mesh_url = m.get("URI")
        if mesh_url is None:
            continue
        # Remove qualifier part, if exists
        cui = mesh_url.split('/')[-1]
        if len(cui) > 10:
            cui = cui[:cui.find('Q')]
        meshes.append(cui)
    # Language
    language = rec.find("Language[@LangType='Primary']")
    entry['issn'] = issn.text if issn is not None else None
    entry['title'] = title.text if title is not None else None
    entry['mesh'] = meshes
    entry['language'] = language.text if language is not None else None
    journals[key.text] = entry
print("{} journals are found".format(len(journals)))
journals_xml = None

# Build JournalMesh Indexes
jmesh2idx = dict()
idx2jmesh = []
for title, data in journals.items():
    for mesh in data['mesh']:
        if mesh not in jmesh2idx:
            jmesh2idx[mesh] = len(jmesh2idx)
            idx2jmesh.append(mesh)

# Read training dataset
print("Parsing training dataset...")
with open(t_file) as f:
    data = json.load(f)
# Connect the concept database
cnx = sqlite3.connect(db_file)
not_found = 0
qmesh2idx = dict()  # Build QueryMesh Index
idx2qmesh = []
qmeshes = dict()  # mesh list by question id
for q in data['questions']:
    concepts = set()
    csr = cnx.cursor()
    csr.execute("SELECT * FROM concepts2 WHERE id='{}'".format(q['id']))
    res = csr.fetchall()
    for c in res:
        # Read MeSHes from the concepts database
        (qid, source, cids, name0, nameN) = c
        if source == 'MetaMap':
            concepts |= set(cids.split(';'))
        else:
            for t in cids.split(';'):
                c_ = t.split(':')
                if len(c_) == 3 and c_[1] == 'MESH':
                    concepts.add(c_[2])
    if len(concepts) == 0:
        not_found += 1
        continue
    else:
        # Add a concept to QueryMesh Indexes
        for c in concepts:
            if c not in qmesh2idx:
                qmesh2idx[c] = len(qmesh2idx)
                idx2qmesh.append(c)
        # Add concepts to qmeshes
        qmeshes[q['id']] = concepts
print("{} out of {} questions are not found in the concepts db"
      "".format(not_found, len(data['questions'])))

# Count the occurrences;
#   1. For each question, find the document sources
#   2. count the journal meshes of the sources w.r.t the query meshes
dist_table = np.zeros((len(qmesh2idx), len(jmesh2idx)), dtype=int)
progress = 0
total = len(data['questions'])
print("Building a mesh distribution table...")


def cb_update_table(rst):
    global progress
    progress += 1
    if rst is None:
        return
    qid, titles = rst
    print('progress: {} / {}\r'.format(progress, total), end='', flush=True)
    for qmesh in qmeshes[qid]:
        for title in titles:
            for jmesh in journals[title]['mesh']:
                dist_table[qmesh2idx[qmesh], jmesh2idx[jmesh]] += 1


def read_journal_meshes(q):
    titles = []
    if q['id'] not in qmeshes:
        return
    for doc_url in q['documents']:
        # Read each from galago corpus
        doc_id = doc_url.split('/')[-1]
        if re.match(r"\d+$", doc_id) is None:
            continue
        p = subprocess.run(['galago', 'doc', '--index={}'.format(galago_idx),
                            '--id=PMID-{}'.format(doc_id)],
                           stdout=subprocess.PIPE )
        body = p.stdout.decode('utf-8')
        # Extract journal source title
        fld = 'MEDLINE_TA'
        start = body.find('<{}>'.format(fld)) + len(fld) + 2
        end = body.find('</{}>'.format(fld))
        if not (end <= start <= len(fld) + 2 and end <= 0):
            title = body[start:end]
        else:
            continue
        if title in journals:
            titles.append(title)
    return q['id'], titles

p = Pool()
for i, q in enumerate(data['questions']):
    p.apply_async(read_journal_meshes, args=(q, ), callback=cb_update_table)
p.close()
p.join()

# Save the results
state = {
    'journals': journals,
    'jmesh2idx': jmesh2idx,
    'idx2jmesh': idx2jmesh,
    'qmesh2idx': qmesh2idx,
    'idx2qmesh': idx2qmesh,
    'dist_table': dist_table
}

print('\n{} journal meshes indexed'.format(len(jmesh2idx)))
print('{} query meshes indexed'.format(len(qmesh2idx)))
print('{}/{} distribution table updated'
      ''.format(np.count_nonzero(dist_table), np.prod(dist_table.shape)))
print('Saving the results...')
if output_overwrite:
    pickle.dump(state, output_file.open('wb'))

