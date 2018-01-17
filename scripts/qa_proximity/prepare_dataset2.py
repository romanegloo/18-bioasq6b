#!/usr/bin/env python3
"""This runs the same as prepare_dataset.py, but this script is specifically
for reading BioASQ-trainingDataset{}.json. The list of questions for a test
dataset is required, which can be extracted from comparing test datasets of
previous years (ref. /code/scripts/galago/extract_q5.py)"""

import os
import json
import spacy
import subprocess
import random
from multiprocessing import Pool
from pathlib import Path, PosixPath
from tqdm import tqdm

# path to files
DATA_DIR = os.path.join(PosixPath(__file__).absolute().parents[3].as_posix(),
                        'data')
IDX_DIR = os.path.join(DATA_DIR, 'galago-medline-full-idx')
INPUT_DIR = os.path.join(DATA_DIR, 'bioasq/train')
OUT_DIR = os.path.join(DATA_DIR, 'qa_prox')

year = 6   # either 5 or 6  (0 for debugging)
train_infile = os.path.join(INPUT_DIR,
                            'BioASQ-trainingDataset{}b.json'.format(year))
test_infile = os.path.join(INPUT_DIR, 'qids_year{}.txt'.format(year-1))
# Consider only the snippets from the allowed sections
sections = ['title', 'abstract']


# define helper functions
def offset_overlaps(offset, length, spans):
    for span in spans:
        if span[0] != 'abstract':  # only abstract
            continue
        if offset >= span[1] and offset <= span[2]:
            return True
        if offset + length >= span[1] and offset + length <= span[2]:
            return True
    return False


def read_irrel_sents(docid, spans):
    irrel_sents = []
    # Read the document. Only use abstract section.
    p = subprocess.run(['galago', 'doc', '--index={}'.format(IDX_DIR),
                        '--id={}'.format(docid)], stdout=subprocess.PIPE)
    doc = p.stdout.decode('utf-8')
    # find the abstract; between <TEXT> and </TEXT>
    start = doc.find('<TEXT>') + len('<TEXT>')
    end = doc.find('</TEXT>')
    if start >= end or start <= len('<TEXT>') or end <= 0:
        return []
    abstract = doc[start:end]

    s_ = nlp(abstract)
    offset = 0
    for sent in s_.sents:
        # check if the sentence overlaps any of the relevant spans
        if not offset_overlaps(offset, len(sent.text), spans):
            irrel_sents.append(sent.text)
                # multiprocessing is not able to pass the results in object type
                # unfortunately, need to re-parse the sentences
        offset += len(sent.text)
    return irrel_sents


def extract_irrel_from_pool(offsets, irrel_sents, num):
    result_list = []
    def log_results(res):
        result_list.extend(res)
    p = Pool()
    for docid, spans in offsets.items():
        p.apply_async(read_irrel_sents, args=(docid, spans),
                      callback=log_results)
    p.close()
    p.join()
    res_ = random.sample(result_list, min(num, len(result_list)))
    irrel_sents.extend(res_)


def extract_irrel_from_others(irrel_sents, num):
    cnt_sents = 0
    sents = []
    while cnt_sents < num:
        # pick a random document
        docid_ = random.randint(1, 26759399)
        p = subprocess.run(['galago', 'doc-name', IDX_DIR + '/names',
                            str(docid_)], stdout=subprocess.PIPE)
        docid = p.stdout.decode('utf-8').strip()
        p = subprocess.run(['galago', 'doc', '--index={}'.format(IDX_DIR),
                            '--id={}'.format(docid)], stdout=subprocess.PIPE)
        doc = p.stdout.decode('utf-8')
        # find the abstract; between <TEXT> and </TEXT>
        start = doc.find('<TEXT>') + len('<TEXT>')
        end = doc.find('</TEXT>')
        if start >= end or start <= len('<TEXT>') or end <= 0:
            continue
        s_ = nlp(doc[start:end])
        for s in s_.sents:
            cnt_sents += 1
            sents.append(s.text)
    irrel_sents.extend(random.sample(sents, num))


def build_dataset(data, questions, mode):
    print("Building {} dataset...".format(mode))
    # for q in tqdm(data['questions']):
    for q in tqdm(data['questions']):
        if 'snippets' not in q:
            continue
        # select questions by mode
        if mode == 'train' and q['id'] in questions:
            continue
        elif mode == 'test' and q['id'] not in questions:
            continue

        # Parse the question first, which is used in common
        q_ = nlp(q['body'])

        # Parse relevant examples
        # ----------------------------------------------------------------------
        rel_offsets = dict()
        cnt_rel = 0
        for s in q['snippets']:
            # Abandon abnormal cases, just continue
            # (Note that datasets of year 1 has section number, which is
            # difficult to maintain offsets from the sequential document text.)
            if s['beginSection'] not in sections or \
                    s['endSection'] not in sections:
                # print('unsupported section type - ({}, {})'
                #       ''.format(s['beginSection'], s['endSection']))
                continue
            if s['beginSection'] != s['endSection']:
                # print('inconsistent sections - ({}, {})'
                #       ''.format(s['beginSection'], s['endSection']))
                continue
            docid = 'PMID-' + s['document'].split('/')[-1]
            # Keep track of the offsets, so that the irrelevant texts can be
            # extracted from the outside of the offset spans.
            if docid not in rel_offsets:
                rel_offsets[docid] = []
            rel_offsets[docid].append((s['beginSection'],
                                       s['offsetInBeginSection'],
                                       s['offsetInEndSection']))

            s_ = nlp(s['text'])
            for _ in s_.sents:
                cnt_rel += 1
                rec = dict()
                rec['qid'] = q['id']
                rec['question'] = [t.text.lower() for t in q_]
                rec['q_pos'] = [t.pos_ for t in q_]
                rec['q_ner'] = [t.ent_type_ for t in q_]
                rec['type'] = q['type']
                rec['label'] = 1
                rec['context'] = [t.text.lower() for t in s_]
                rec['pos'] = [t.pos_ for t in s_]
                rec['ner'] = [t.ent_type_ for t in s_]
                with open(os.path.join(OUT_DIR, mode,
                                       'rel-t{}.txt'.format(year)), 'a') as f:
                    f.write(json.dumps(rec) + '\n')

        # Parse irrelevant examples
        # ----------------------------------------------------------------------
        # : We need to generate the same number of irrelevant examples; 50%
        # from the same relevant document pool, and the rest 50% from the
        # randomly selected pubmed documents (which assumed to be irrelevant
        # by chance)
        irrel_sents = []
        extract_irrel_from_pool(rel_offsets, irrel_sents, int(cnt_rel/2))
        extract_irrel_from_others(irrel_sents, cnt_rel - len(irrel_sents))
        # # Sample the same number of irrelevant examples
        for sent in irrel_sents:
            s_ = nlp(sent)
            rec = dict()
            rec['qid'] = q['id']
            rec['question'] = [t.text.lower() for t in q_]
            rec['q_pos'] = [t.pos_ for t in q_]
            rec['q_ner'] = [t.ent_type_ for t in q_]
            rec['type'] = q['type']
            rec['label'] = 0
            rec['context'] = [t.text.lower() for t in s_]
            rec['pos'] = [t.pos_ for t in s_]
            rec['ner'] = [t.ent_type_ for t in s_]
            with open(os.path.join(OUT_DIR, mode,
                                   'irrel-t{}.txt'.format(year)), 'a+') as f:
                f.write(json.dumps(rec) + '\n')  # write out


if __name__ == '__main__':
    # load spacy nlp
    print("Loading SpaCy 'en' module...")
    nlp = spacy.load('en')

    # create directories, if not exist
    for d in ['train', 'test']:
        dir = os.path.join(OUT_DIR, d)
        if not os.path.exists(dir):
            Path(dir).mkdir(parents=True)
        rel_file = os.path.join(OUT_DIR, d, 'rel-t{}.txt'.format(year))
        irrel_file = os.path.join(OUT_DIR, d, 'irrel-t{}.txt'.format(year))
        if os.path.exists(rel_file):
            os.remove(rel_file)
        if os.path.exists(irrel_file):
            os.remove(irrel_file)

    # read from test/training data files
    print('reading from bioasq training file...({})'.format(train_infile))
    with open(train_infile) as f:
        data = json.load(f)
    print('reading the test questions...({})'.format(test_infile))
    questions = open(test_infile).read().split('\n')

    for mode in ['train', 'test']:
        build_dataset(data, questions, mode)

