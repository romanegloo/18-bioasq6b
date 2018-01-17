#!/usr/bin/env python3
"""Reads bioasq test datasets which contain pairs of questions and relevant
text snippets. The resulting datasets will be used for training a neural
model that classifies the relevance of a question to the candidate text
snippets"""

import os
import json
import spacy
import subprocess
import random
from multiprocessing import Pool
import shutil
from pathlib import Path

# path to files
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')
IDX_DIR = os.path.join(DATA_DIR, 'galago-medline-full-idx')
TEST_DIR = os.path.join(DATA_DIR, 'bioasq/test')
OUT_DIR = os.path.join(DATA_DIR, 'qa_prox')

TRAIN_FROM = [2,3]   # year 2 and 3 only search over abstracts and titles
TEST_FROM = [4]
SECTIONS = ['title', 'abstract']


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

def extract_irrel_ex(docid, spans):
    irrel_sents = []
    print('reading docs for generating irrelevant examples...{}'
          ''.format(docid), end='\r')
    # read the abstract of the doc
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
    for sent in s_.sents:  # split on sentences
        # check relevant offset spans
        if not offset_overlaps(offset, len(sent.text), spans):
            irrel_sents.append(sent.text)
        offset += len(sent.text)
    return irrel_sents


def build_dataset(questions, mode):
    for cid, q in enumerate(questions):
        print('processing: qid-{} [{}/{}]'
              ''.format(q['id'], cid+1, len(questions)) + ' '*20)
        if 'snippets' not in q:
            continue

        # Parse the question first, which is used in common
        q_ = nlp(q['body'])

        # Parse relevant examples
        rel_offsets= {}
        cnt_rel = 0
        for s in q['snippets']:
            # Abandon abnormal cases, just continue
            if s['beginSection'] not in SECTIONS or \
                    s['endSection'] not in SECTIONS:
                print('unsupported section type - ({}, {})'
                      ''.format(s['beginSection'], s['endSection'] ))
                continue
            if s['beginSection'] != s['endSection']:
                print('inconsistent sections - ({}, {})'
                      ''.format(s['beginSection'], s['endSection'] ))
                continue
            docid = 'PMID-' + s['document'].split('/')[-1]
            # Keep track of the offsets, so that the irrelevant texts can be
            # extracted from the outside of the offset regions.
            if docid not in rel_offsets:
                rel_offsets[docid] = []
            rel_offsets[docid].append((s['beginSection'],
                                       s['offsetInBeginSection'],
                                       s['offsetInEndSection']))

            s_ = nlp(s['text'])
            for sent in s_.sents:
                cnt_rel += 1
                rec = {}
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
                                       'rel-test{}.txt'.format( TEST_FROM[0])),
                          'a') as f:
                    f.write(json.dumps(rec) + '\n')

        # Generate irrelevant examples
        # ----------------------------------------------------------------------
        irrel_sents = []  # generate irrelevant sentence list
        p = Pool(20)
        results = [p.apply_async(extract_irrel_ex, args=(docid, spans))
                   for docid, spans in rel_offsets.items()]
        for rst in results:
            irrel_sents.extend(rst.get())
        p.close()

        # Sample the same number of irrelevant examples
        irrel_sents = random.sample(irrel_sents, min(cnt_rel, len(irrel_sents)))
        for sent in irrel_sents:
            s_ = nlp(sent)
            rec = {}
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
                                   'irrel-test{}.txt'.format(TEST_FROM[0])),
                      'a+') as f:
                f.write(json.dumps(rec) + '\n')  # write out


if __name__ == '__main__':
    # load spacy nlp
    nlp = spacy.load('en')

    # start from scratch
    # todo. change it to maintain the existing files
    if os.path.exists(OUT_DIR):
        shutil.rmtree(os.path.join(OUT_DIR, 'train'))
        shutil.rmtree(os.path.join(OUT_DIR, 'test'))
    Path(os.path.join(OUT_DIR, 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUT_DIR, 'test')).mkdir(parents=True, exist_ok=True)

    # read from test files (phaseB_*b_0*.json), for training dataset
    print('reading from bioasq test files...({})'.format(TEST_DIR))
    data_from = [TRAIN_FROM, TEST_FROM]
    for data_idx in range(2):
        mode = 'train' if data_idx == 0 else 'test'
        print('building dataset for {}'.format(mode))
        questions = []
        for year in data_from[data_idx]:
            for batch in range(1, 6):
                if year == 1 and batch >= 4:
                    continue  # the first year has only 3 batches
                batch_file = os.path.join(TEST_DIR,
                                          "phaseB_{}b_0{}.json".format(year, batch))
                with open(batch_file) as f:
                    data = json.load(f)
                    questions.extend(data['questions'])
        build_dataset(questions, mode)
