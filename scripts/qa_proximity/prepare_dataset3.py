"""A variant preprocessing script from prepare_dataset2.py. This script writes
the datasets for QA_sim model that predicts the answer span (text
snippets), instead of classifying sentences in the similarity to the question.
The fields of each question-answer span includes [lemma, question, ner, pos,
offsets, answers, document, qlemma, qid, idf, qtype]."""

import os
from pathlib import Path
import spacy
import subprocess
import json
import pickle
import math
from collections import OrderedDict
import re
import random
import logging
import multiprocessing
from multiprocessing import Pool


def parse_qa_snippets(qa):
    """Reads qa pairs, generate examples from the text snippets"""
    doc_record = dict()
    ordered_keys = ['qid', 'qtype', 'question', 'docid', 'qlemma', 'context',
                    'lemma', 'ner', 'pos', 'offsets', 'idf', 'answers']
    sections = ['abstract', 'title', 'sections.0']
    q_ = nlp(qa['body'])
    for s in qa['snippets']:
        rec = OrderedDict.fromkeys(ordered_keys)
        # Ignore unknown sections, just continue
        if s['beginSection'] not in sections:
            continue
        if s['beginSection'] != s['endSection']:
            continue
        docid = 'PMID-' + s['document'].split('/')[-1]
        if docid not in doc_record:
            # Fill data in document level and read document
            p = subprocess.run(['galago', 'doc', '--index={}'.format(idx_dir),
                               '--id={}'.format(docid)], stdout=subprocess.PIPE)
            doc = p.stdout.decode('utf-8')
            rec['docid'] = docid
            rec['question'] = [t.text.lower() for t in q_]
            rec['qlemma'] = [t.lemma_ for t in q_]
            rec['qid'] = qa['id']
            rec['qtype'] = qa['type']
            # Find the title and abstract
            context = {}
            for f in ['TEXT', 'TITLE']:
                len_ = len(f) + 2
                start = doc.find('<{}>'.format(f)) + len_
                end = doc.find('</{}>'.format(f))
                key = 'abstract' if f == 'TEXT' else 'title'
                if not (end <= start <= len_ and end <= 0):
                    context[key] = doc[start:end]
            # Some titles are enclosed in brackets
            context['title'] = re.sub(r'[\[\]]', '', context['title'])
            # Parse title and abstract
            title_ = nlp(context['title'])
            abstract_ = nlp(context['abstract'])
            for k in ['context', 'lemma', 'ner', 'pos', 'offsets', 'idf']:
                rec[k] = [[]] * 2
            for i, text in enumerate([title_, abstract_]):
                rec['context'][i] = [t.text.lower() for t in text]
                rec['lemma'][i] = [t.lemma_ for t in text]
                rec['ner'][i] = [t.ent_type_ for t in text]
                rec['pos'][i] = [t.pos_ for t in text]
                rec['idf'][i] = [idf[t.text.lower()] if t.text.lower() in idf
                                 else 0 for t in text]
                offset_start = 0
                rec['offsets'][i] = []
                for token in text:
                    rec['offsets'][i].append((offset_start, len(token)))
                    offset_start += len(token) + 1
            rec['answers'] = []
            doc_record[docid] = rec
        doc_record[docid]['answers'].append((s['beginSection'],
                                             s['offsetInBeginSection'],
                                             s['offsetInEndSection']))
    # Convert answer offsets into tokens
    logger.info('1')
    for k, doc in doc_record.items():
        logger.info('2')
        answers = []
        for span in doc['answers']:
            logger.info('3')
            begin_ = 0
            end_ = 0
            section = 0 if span[0] == 'title' else 1
            idx_cont = 0
            for i, t in enumerate(doc['offsets'][section]):
                logger.info('4')
                if t[0] <= span[1] <= sum(t):
                    begin_ = i
                    idx_cont = i
                    break
            for i, t in enumerate(doc['offsets'][section][idx_cont+1:]):
                logger.info('5')
                if t[0] <= span[2] <= sum(t):
                    end_ = i + idx_cont + 1
                    break
            logger.info('6')
            answers.append((span[0], begin_, end_))
        logger.info('7')
        doc['answers'] = answers

    return doc_record, qa['id']


def parse_qa(qa, docnum_max):
    """Reads qa pairs, random sample documents from the entire pool,
    and generate negative examples"""
    doc_record = dict()
    ordered_keys = ['qid', 'qtype', 'question', 'docid', 'qlemma', 'context',
                    'lemma', 'ner', 'pos', 'offsets', 'idf', 'answers']
    q_ = nlp(qa['body'])
    # Count documents that has text snippets, and sample 1/2 of the size
    # random documents
    numdocs = int(len(set([s['document'] for s in qa['snippets']])) / 2 + .5)
    sample = [random.randint(1, docnum_max) for _ in range(numdocs)]
    for docid in sample:
        rec = OrderedDict.fromkeys(ordered_keys)
        # Get PMID from document sequence number
        p = subprocess.run(['galago', 'doc-name', (idx_dir/'names').as_posix(),
                            str(docid)], stdout=subprocess.PIPE)
        pmid = p.stdout.decode('utf-8').strip()
        # Read document and fill the data in document level
        p = subprocess.run(['galago', 'doc',
                            '--index={}'.format(idx_dir.as_posix()),
                            '--id={}'.format(pmid)], stdout=subprocess.PIPE)
        doc = p.stdout.decode('utf-8')
        rec['docid'] = pmid
        rec['question'] = [t.text.lower() for t in q_]
        rec['qlemma'] = [t.lemma_ for t in q_]
        rec['qid'] = qa['id']
        rec['qtype'] = qa['type']
        # Find the title and abstract
        context = {}
        for f in ['TEXT', 'TITLE']:
            len_ = len(f) + 2
            start = doc.find('<{}>'.format(f)) + len_
            end = doc.find('</{}>'.format(f))
            key = 'abstract' if f == 'TEXT' else 'title'
            if not (end <= start <= len_ and end <= 0):
                context[key] = doc[start:end]
            # Some titles are enclosed in brackets
        context['title'] = re.sub(r'[\[\]]', '', context['title'])
        # Parse title and abstract
        title_ = nlp(context['title'])
        abstract_ = nlp(context['abstract'])
        for k in ['context', 'lemma', 'ner', 'pos', 'offsets', 'idf']:
            rec[k] = [[]] * 2
        for i, text in enumerate([title_, abstract_]):
            rec['context'][i] = [t.text.lower() for t in text]
            rec['lemma'][i] = [t.lemma_ for t in text]
            rec['ner'][i] = [t.ent_type_ for t in text]
            rec['pos'][i] = [t.pos_ for t in text]
            rec['idf'][i] = [idf[t.text.lower()] if t.text.lower() in idf
                             else 0 for t in text]
            offset_start = 0
            for token in text:
                rec['offsets'][i].append((offset_start, len(token)))
                offset_start += len(token) + 1
        rec['answers'] = []
        doc_record[docid] = rec
    return doc_record, qa['id']


def write_examples(rst):
    with (output_dir / mode / 'examples_y6.txt').open('a') as f:
        for k, v in rst[0].items():
            f.write(json.dumps(v) + '\n')
    print('{} examples added (qid: {})'.format(len(rst[0]), rst[1]))


def build_dataset(data, dev_pairs):
    # Select pairs in mode
    if mode == 'dev':
        pairs = [q for q in data['questions'] if q['id'] in dev_pairs]
    else:
        pairs = [q for q in data['questions'] if q['id'] not in dev_pairs]

    print("Adding pos examples into {} dataset ({} pairs)..."
          ''.format(mode, len(pairs)))
    p = Pool()
    for qa in pairs:
        p.apply_async(parse_qa_snippets, args=(qa,), callback=write_examples)
    p.close()
    p.join()

    # below is commented out: no way to train with unlabelled documents
    # print("Adding neg examples into {} dataset ({} pairs)..."
    #       ''.format(mode, len(pairs)))
    # Add negative examples (from irrelevant documents)
    # p = Pool()
    # for qa in pairs:
    #     p.apply_async(parse_qa, args=(qa, docnum_max), callback=write_examples)
    # p.close()
    # p.join()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Set paths (assumes all required files are prepared and accessible)
    data_dir = Path(__file__).absolute().parents[3] / 'data'
    idx_dir = data_dir / 'galago-medline-full-idx'
    input_dir = data_dir / 'bioasq/train'
    output_dir = data_dir / 'qa_prox'
    train_file = (input_dir / 'BioASQ-trainingDataset6b.json').as_posix()
    test_file = (input_dir / 'qids_year5.txt').as_posix()
    termstats_file = (idx_dir / 'termstats.tsv').as_posix()

    # Load SpaCy NLP module
    print("Loading SpaCy 'en' module...")
    nlp = spacy.load('en')

    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.ERROR)

    # Read the manifest from the Galago index to read the total no. of docs
    p = subprocess.run(['galago', 'dump-index-manifest',
                        (idx_dir/'corpus').as_posix()],
                       stdout=subprocess.PIPE)
    manifest = json.loads(p.stdout.decode('utf-8'))
    docnum_max = manifest['keyCount']

    # Run galago dump-term-stats, if termstats file does not exist
    if not os.path.isfile(termstats_file):
        print('Saving termstats of the galago index postings...')
        f = open(termstats_file, 'w')
        p = subprocess.run(['galago', 'dump-term-stats',
                            os.path.join(idx_dir, 'postings')], stdout=f)
        f.close()

    # Write out idf values by terms
    if not (output_dir/'idf.p').is_file():
        print('Reading term-doc frequencies from the Galago index')
        idf = dict()
        with open(termstats_file) as f:
            for i, l in enumerate(f):
                t = l.split('\t')
                if len(t) < 3:
                    continue
                idf[str(' '.join(t[:-2]))] = \
                    math.log(docnum_max / (1 + int(t[-1])))
        pickle.dump(idf, (output_dir/'idf.p').open('wb'))
    else:  # Read in
        print('Reading idf values by terms...')
        with (output_dir/'idf.p').open('rb') as f:
            idf = pickle.load(f)

    # Read from test/training data files
    print('Reading from bioasq training file...({})'.format(train_file))
    with open(train_file) as f:
        data = json.load(f)
    print('Reading the test questions...({})'.format(test_file))
    questions = open(test_file).read().split('\n')

    # Create directories, if not exist
    for d in ['train', 'dev']:
        if not (output_dir / d).exists():
            print('Creating directories for datasets.')
            (output_dir / d).mkdir(parents=True)
        output_file = (output_dir / d / 'examples_y6.txt')
        if output_file.exists():
            print('Removing existing file ({}/examples_y6.txt)'.format(d))
            output_file.unlink()

    mode = None
    for mode in ['train', 'dev']:
        build_dataset(data, questions)
