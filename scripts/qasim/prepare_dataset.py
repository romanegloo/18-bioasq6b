#!/usr/bin/env python3
"""This runs the same as prepare_dataset.py, but this script is specifically
for reading BioASQ-trainingDataset{}.json. The list of questions for a test
dataset is required, which can be extracted from comparing test datasets of
previous years (ref. /code/scripts/galago/extract_q5.py)"""

import os
import glob
import json
import spacy
import subprocess
import random
from multiprocessing import Pool
from pathlib import PosixPath
from tqdm import tqdm
import pickle
import math


# define helper functions
def offset_overlaps(offset, length, spans):
    for span in spans:
        if span[0] != 'abstract':  # only abstract
            continue
        if span[1] <= offset <= span[2]:
            return True
        if span[1] <= offset + length <= span[2]:
            return True
    return False


def read_irrel_sents_pool(docid, spans):
    irrel_sents = []
    # Read the document. Only use abstract section.
    p = subprocess.run(['galago', 'doc',
                        '--index={}'.format(paths['galago_index']),
                        '--id={}'.format(docid)], stdout=subprocess.PIPE)
    doc = p.stdout.decode('utf-8')
    # find the abstract; between <TEXT> and </TEXT>
    start = doc.find('<TEXT>') + len('<TEXT>')
    end = doc.find('</TEXT>')
    if start >= end or start <= len('<TEXT>') or end <= 0:
        return []
    text = nlp(doc[start:end])
    offset = 0
    for sent in text.sents:
        # check if the sentence overlaps any of the relevant spans
        if not offset_overlaps(offset, len(sent.text), spans):
            rec = dict()
            rec['context'] = [t.text.lower() for t in sent]
            rec['pos'] = [t.pos_ for t in sent]
            rec['ner'] = [t.ent_type_ for t in sent]
            irrel_sents.append(rec)
        offset += len(sent.text)
    return irrel_sents


def extract_irrel_from_pool(offsets, num):
    results = []
    p = Pool()
    for docid, spans in offsets.items():
        p.apply_async(read_irrel_sents_pool, args=(docid, spans),
                      callback=lambda res: results.extend(res))
    p.close()
    p.join()
    return random.sample(results, min(num, len(results)))


# def read_irrel_sents_others(docno):
#     p = subprocess.run(['galago', 'doc-name',
#                         paths['galago_index'] + '/names',
#                         str(docno)], stdout=subprocess.PIPE)
#     docid = p.stdout.decode('utf-8').strip()
#     p = subprocess.run(['galago', 'doc',
#                         '--index={}'.format(paths['galago_index']),
#                         '--id={}'.format(docid)], stdout=subprocess.PIPE)
#     doc = p.stdout.decode('utf-8')
#     # find the abstract; between <TEXT> and </TEXT>
#     start = doc.find('<TEXT>') + len('<TEXT>')
#     end = doc.find('</TEXT>')
#     if start >= end or start <= len('<TEXT>') or end <= 0:
#         return []
#     text = nlp(doc[start:end])
#     sents = [s.text for s in text.sents]
#     return [random.choice(sents)]


def extract_irrel_from_others(num):
    cnt_sents = 0
    results = []
    while cnt_sents < num:
        # pick a random document
        docid_ = random.randint(1, total_docs)
        p = subprocess.run(['galago', 'doc-name',
                            paths['galago_index'] + '/names',
                            str(docid_)], stdout=subprocess.PIPE)
        docid = p.stdout.decode('utf-8').strip()
        p = subprocess.run(['galago', 'doc',
                            '--index={}'.format(paths['galago_index']),
                            '--id={}'.format(docid)], stdout=subprocess.PIPE)
        doc = p.stdout.decode('utf-8')
        # find the abstract; between <TEXT> and </TEXT>
        start = doc.find('<TEXT>') + len('<TEXT>')
        end = doc.find('</TEXT>')
        if start >= end or start <= len('<TEXT>') or end <= 0:
            continue
        s_ = nlp(doc[start:end])
        for sent in s_.sents:
            cnt_sents += 1
            rec = dict()
            rec['context'] = [t.text.lower() for t in sent]
            rec['pos'] = [t.pos_ for t in sent]
            rec['ner'] = [t.ent_type_ for t in sent]
            results.append(rec)
    return random.sample(results, num)


def build_dataset(data, questions, year):
    """
    Build datasets for training/testing of QAsim model by processing BioAsq
    training datasets.
    Proportions:
        - 20% of relevant examples and 80% of irrelevant examples.
        - out of 80% irrelevant examples, half are from the relevant documents
          at most

    :param data:
    :param questions:
    :param year:
    :return:
    """
    # Consider only the text snippets from the 'title' and 'abstract'
    sections = ['title', 'abstract']
    # Split into train and test
    data_for = ['train', 'test']
    collections = [[], []]
    for q in data['questions']:
        if 'snippets' not in q:
            continue
        if q['id'] in questions:
            collections[1].append(q)
        else:
            collections[0].append(q)

    for i, type in enumerate(data_for):
        print('Building a dataset for {}...'.format(type))
        for q in tqdm(collections[i]):
            # Process the question sentence using SpaCy parser
            q_ = nlp(q['body'])
            # Parse relevant examples
            # ------------------------------------------------------------------
            rel_offsets = dict()
            examples = []
            for snippet in q['snippets']:
                # Just ignore abnormal cases, which are difficult to process
                if snippet['beginSection'] not in sections or \
                        snippet['endSection'] not in sections:
                    continue
                if snippet['beginSection'] != snippet['endSection']:
                    continue
                docid = 'PMID-' + snippet['document'].split('/')[-1]
                if docid not in rel_offsets:
                    rel_offsets[docid] = []
                # Keep track of the offsets, so that the irrelevant texts can be
                # extracted from the outside of the offset spans.
                rel_offsets[docid].append((snippet['beginSection'],
                                           snippet['offsetInBeginSection'],
                                           snippet['offsetInEndSection']))
                text = nlp(snippet['text'].strip())
                for sent in text.sents:
                    if len(sent) <= 3:
                        continue
                    rec = dict()
                    rec['qid'] = q['id']
                    rec['question'] = [t.text.lower() for t in q_]
                    rec['q_pos'] = [t.pos_ for t in q_]
                    rec['q_ner'] = [t.ent_type_ for t in q_]
                    rec['type'] = q['type']
                    rec['label'] = 1
                    rec['context'] = [t.text.lower() for t in sent]
                    rec['pos'] = [t.pos_ for t in sent]
                    rec['ner'] = [t.ent_type_ for t in sent]
                    examples.append(rec)

            # Parse irrelevant examples
            # ------------------------------------------------------------------
            # : 50% of the irrelevant examples are chosen from the relevant
            # document pool and the rest 50% are chosen from randomly selected
            # pubmed document pool (assuming that the chance of the documents
            # being relevant is extremely low)

            num_to_extract = len(examples) * 2
            examples_a = extract_irrel_from_pool(rel_offsets, num_to_extract)
            num_to_extract = len(examples) * 4 - len(examples_a)
            examples_b = extract_irrel_from_others(num_to_extract)
            for sent in examples_a + examples_b:
                # s_ = nlp(sent)
                rec = dict()
                rec['qid'] = q['id']
                rec['question'] = [t.text.lower() for t in q_]
                rec['q_pos'] = [t.pos_ for t in q_]
                rec['q_ner'] = [t.ent_type_ for t in q_]
                rec['type'] = q['type']
                rec['label'] = 0
                rec['context'] = sent['context']
                rec['pos'] = sent['pos']
                rec['ner'] = sent['ner']
                examples.append(rec)
            output_file = os.path.join(output_dir,
                                       'examples-y{}-{}.txt'.format(year, type))
            with open(output_file, 'a') as f:
                for entry in examples:
                    f.write(json.dumps(entry) + '\n')


def build_idf_dictionary():
    idf_file = os.path.join(paths['data_dir'], 'idf.pkl')
    if os.path.exists(idf_file):
        return
    # Read the manifest from the Galago index in order to get the total
    # number of documents indexed
    p = subprocess.run(['galago', 'dump-index-manifest',
                        os.path.join(paths['galago_index'], 'corpus')],
                       stdout=subprocess.PIPE)
    manifest = json.loads(p.stdout.decode('utf-8'))
    global total_docs
    total_docs = manifest['keyCount']

    # Read term frequency list
    print('Reading the term-doc frequencies from the Galago index...')
    idf = dict()
    termstats_file = os.path.join(paths['galago_index'], 'termstats.tsv')
    with open(termstats_file) as f:
        for i, l in enumerate(f):
            t = l.split('\t')
            if len(t) < 3:
                continue
            idf[str(' '.join(t[:-2]))] = math.log(total_docs / (1 + int(t[-1])))
    # Write out idf values by terms
    pickle.dump(idf, open(idf_file, 'wb'))
    print('IDF dictionary is created')


if __name__ == '__main__':
    # Set defaults
    years = [5, 6]  # build datasets for both year 5 and 6
    paths = dict()
    paths['root_dir'] = PosixPath(__file__).absolute().parents[2].as_posix()
    paths['data_dir'] = os.path.join(paths['root_dir'], 'data')
    paths['galago_index'] = os.path.join(paths['data_dir'],
                                         'galago-medline-full-idx')

    # Load SpaCy NLP module
    print("Loading SpaCy 'en' module...")
    nlp = spacy.load('en')

    total_docs = 26759399
    build_idf_dictionary()

    # Create directories, if not exist
    output_dir = os.path.join(paths['data_dir'], 'bioasq_processed')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for f in glob.glob(os.path.join(output_dir, 'examples-*')):
        os.remove(f)
    for y in years:
        # train_infile contains all the questions/answers prior to the year
        # test_infile is a list question ids that will be used for test
        train_infile = os.path.join(
            paths['data_dir'],
            'bioasq/train/BioASQ-trainingDataset{}b.json'.format(y))
        test_infile = os.path.join(
            paths['data_dir'], 'bioasq/train/qids_year{}.txt'.format(y - 1))
        # Read from test/training data files
        with open(train_infile) as f:
            print('Reading from bioasq training file...({})'
                  ''.format(train_infile))
            data = json.load(f)
        print('Reading the list of test questions...({})'.format(test_infile))
        questions = [line.rstrip() for line in open(test_infile)]


        build_dataset(data, questions, y)
