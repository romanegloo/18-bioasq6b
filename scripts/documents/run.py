#!/usr/bin/env python3
"""User script that can be run for both optimization and evaluation on BioASQ
QA datasets"""

import argparse
import os
import sys
import json
import math
from datetime import datetime
import logging, coloredlogs
import pickle
import spacy

from BioAsq6B.retriever import GalagoSearch
from BioAsq6B import reranker, PATHS
from BioAsq6B.common import AverageMeter
from BioAsq6B.data_services import ConceptRetriever


def init():
    """Set default values and initialize components"""
    qasim_ranker = journal_ranker = semmeddb_ranker = None
    # --------------------------------------------------------------------------
    # Set default options
    # --------------------------------------------------------------------------
    if args.qasim_model is not None:
        PATHS['qasim_model'] = os.path.join(PATHS['data_dir'],
                                      'qa_prox/var/{}'.format(args.qasim_model))
    # Get a list of rerankers from the score weights parameter
    if args.score_weights is not None:
        keys = set()
        for scheme in args.score_weights.split(','):
            if scheme.startswith('qasim:'):
                keys.add('qasim')
            if scheme.startswith('journal:'):
                keys.add('journal')
            if scheme.startswith('semmeddb'):
                keys.add('semmeddb')
        args.rerank = list(keys)
    if args.run_id is None:
        args.run_id = datetime.today().strftime("%b%d-%H%M")
    PATHS['log_file'] = os.path.join(PATHS['runs_dir'], args.run_id + '.log')

    # --------------------------------------------------------------------------
    # Configure a logger
    # --------------------------------------------------------------------------
    coloredlogs.install(
        level='DEBUG',
        fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s"
    )
    if not args.verbose:
        fmt = logging.Formatter(
            '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
        file = logging.FileHandler(PATHS['log_file'])
        file.setFormatter(fmt)
        logger.addHandler(file)
        print("writing output in {}".format(PATHS['log_file']))
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # --------------------------------------------------------------------------
    # Read datasets (question/answer pairs)
    # --------------------------------------------------------------------------
    # Reading datasets.
    questions = []
    if args.mode != 'dry':
        if args.batch is None:  # If batch is not specified, read all
            args.batch = '1,2,3,4,5'
        batches = args.batch.split(',')
        for b in batches:
            batch_file = 'phaseB_{}b_0{}.json'.format(args.year, b)
            logger.info('Reading test dataset: {}'.format(batch_file))
            batch_file = os.path.join(PATHS['test_dir'], batch_file)
            with open(batch_file) as f:
                data = json.load(f)
                if args.qid is not None:
                    for q in data['questions']:
                        if args.qid == q['id']:
                            questions.append([q])
                else:
                    if args.debug:
                        questions.append(data['questions'][:3])
                    else:
                        questions.append(data['questions'])
        if args.qid is not None:
            args.batch = '1'
        logger.info('Testing pairs: {}'.format([len(b) for b in questions]))
    elif args.mode == 'dry':
        if args.dryrun_file is None:
            # Assuming a specific year and batch is provided
            args.dryrun_file = \
                os.path.join(PATHS['test_dir'],
                             'phaseA_{}b_0{}.json'.format(args.year,
                                                          args.batch))
        with open(args.dryrun_file) as f:
            data = json.load(f)
            if args.debug:
                questions.append(data['questions'][:3])
            else:
                questions.append(data['questions'])
        logger.info('Dryrun pairs: {}'.format([len(b) for b in questions]))

    # --------------------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------------------
    idf = pickle.load(open(PATHS['idf_file'], 'rb'))
    logger.info('{} idf scores loaded'.format(len(idf)))
    logger.info('Loading Spacy parser...')
    nlp = spacy.load('en')
    logger.info('initializing retriever...')
    doc_ranker = GalagoSearch(args, nlp=nlp, idf=idf)

    # QaSim ranker
    if 'qasim' in args.rerank:
        logger.info('initializing QaSim-ranker...')
        qasim_ranker = reranker.RerankQaSim(args, nlp)
        if args.print_parameters:
            from BioAsq6B.QaSimSent import utils
            model_summary = utils.torch_summarize(qasim_ranker.predictor.model)
            logger.info(model_summary)
        # Check if the model needs to load idf data
        if 'idf' in qasim_ranker.predictor.model.conf['features']:
            qasim_ranker.predictor.idf = idf

    # Journal ranker
    if 'journal' in args.rerank:
        logger.info('initializing Journal-ranker...')
        journal_ranker = reranker.RerankerJournal()

    # SemMedDB ranker
    if 'semmeddb' in args.rerank:
        logger.info('initializing SemMedDB-ranker...')
        semmeddb_ranker = reranker.RerankerSemMedDB()

    return questions, doc_ranker, qasim_ranker, journal_ranker, semmeddb_ranker


def add_arguments(parser):
    """Define parameters with user provided arguments"""
    # Runtime Settings
    runtime = parser.add_argument_group('Runtime Settings')
    runtime.add_argument('--mode', type=str, default='test',
                         choices=['test', 'dry'],
                         help='Run mode; dry run returns an output file')
    runtime.add_argument('--run-id', type=str, default=None,
                         help='Identifiable name for each run')
    runtime.add_argument('-y', '--year', type=int, default=6,
                         choices=[3, 4, 5, 6],
                         help='Specify the year of the dataset with which an '
                              'evaluation will be done')
    runtime.add_argument('-b', '--batch', type=str,
                         help='Specify the batch number to be tested')
    runtime.add_argument('-q', '--qid', type=str, default=None,
                        help="One question ID to evaluate on")
    runtime.add_argument('--random-seed', type=int, default=12345,
                         help='set a random seed for sampling operations')
    runtime.add_argument('-v', '--verbose', action='store_true',
                         help='Verbose, print out without writing on a logfile')
    runtime.add_argument('--dryrun-file', type=str,
                         help='dry-run file for phaseA')
    runtime.add_argument('--update-concepts', action='store_true',
                         help='Get concepts from data services, and update '
                              'the existing concepts database')
    runtime.add_argument('--update-word-embeddings', action='store_true',
                         help='Update word embeddings for QAsim model; Adding '
                              'unseen words')
    runtime.add_argument('--output-scores', action='store_true',
                         help='Stores a JSON file of all the scores')
    runtime.add_argument('--debug', action='store_true')
    # Retriever Settings
    retriever = parser.add_argument_group('Retriever Settings')
    retriever.add_argument('--galago-weights', type=str, default=None,
                           help='The weights used in galago query statements')
    retriever.add_argument('--ndocs', type=int, default=30,
                           help='Number of document to retrieve')
    # Reranker settings
    reranker = parser.add_argument_group('Reranker Settings')
    reranker.add_argument('--score-weights', type=str, default="retrieval:1",
                          help='comma separated weights for score fusion; ex. '
                               '"retrieval:0.7,qasim:0.15,journal:0.15"')
    reranker.add_argument('--score-fusion', type=str, default='weighted_sum',
                          choices=['weighted_sum', 'rrf'],
                          help='Score fusion method')
    reranker.add_argument('--query-model', type=str, default='sdm',
                          choices=['baseline', 'sdm', 'sdm_mesh', 'rm3'],
                          help='document retrieval model')
    # Model Architecture: model specific options
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--qasim-model', type=str, default=None,
                        help='Path to a QA_Similarity model')
    model.add_argument('--print-parameters', action='store_true',
                       help='Print out model parameters')


def write_result_articles(res, stats=None, seq=None):
    """Write the results with performance measures"""
    if seq is not None:
        logger.info('=== {} / {} ==='.format(*seq))

    res.write_result(printout=args.verbose, stats=stats)
    if len(res.unseen_words) > 0:
        global unseen_words
        unseen_words |= res.unseen_words
    return


def add_results(res, articles=None, seq=None):
    if seq is not None:
        logger.info('=== {} / {} ==='.format(*seq))
    articles[res.query['id']] = res
    res.write_result(printout=args.verbose)
    if len(res.unseen_words) > 0:
        global unseen_words
        unseen_words |= res.unseen_words
    return


def query(q):
    """run retrieval procedure (optionally rerank) of one question and return
    the result"""
    ranked_docs = doc_ranker.closest_docs(q, k=args.ndocs)
    if 'qasim' in args.rerank or 'journal' in args.rerank:
        # After retrieval, read documents
        if args.verbose:
            logger.info('Reading documents for later use...')
        ranked_docs.read_doc_text()
    if 'qasim' in args.rerank:
        qasim_ranker.get_qasim_scores(ranked_docs)
        ranked_docs.unseen_words = qasim_ranker.predictor.add_words
    if 'journal' in args.rerank:
        journal_ranker.get_journal_scores(ranked_docs)
    if 'semmeddb' in args.rerank:
        semmeddb_ranker.get_semmeddb_scores(ranked_docs)

    ranked_docs.merge_scores(args.score_weights)
    return ranked_docs


def save_results_test(lstRankedDocs, year=None, batch=None):
    """Return format:
    {"questions": [
        {
            "qid": id,
            "body": question body,
            "year": year,
            "batch": batch,
            "ranked_docs": [docid1, docid2, ..., docidn],
            "relevancy": [1, 1, 0, 1, ..., 0]
            "scores": {
                "retrieval": [s1, s2, s3, ..., sn],
                "qasim": [s1', s2', s3', ..., sn'],
                "journal": [ ... ],
                "semmeddb": [ ... ]
            }
        }
    ]}
    """
    questions = []
    for res in lstRankedDocs:
        record = dict()
        record['qid'] = res.query['id']
        record['qbody'] = res.query['body']
        record['year'] = year
        record['batch'] = batch
        record['ranked_docs'] = res.rankings
        record['relevancy'] = [1 if d in res.expected_docs else 0
                               for d in res.rankings]
        record['scores'] = res.scores
        questions.append(record)
    filename = "scores-{}_{}-{}.json".format(year, batch, args.run_id)
    score_file = os.path.join(PATHS['runs_dir'], filename)
    json.dump(questions, open(score_file, 'w'))
    logger.info("Saving the scores [{}]".format(score_file))

def save_results_dry(questions, results):
    """return format:
    {"questions":[
        {
            "type": type,
            "body": body,
            "id": id,
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/1234567"
                , ...
            ],
            "snippets": [
                {
                    "document": "http://www.ncbi.nlm.nih.gov/pubmed/1234567",
                    "text": text,
                    "offsetInBeginSection": 559,
                    "offsetInEndSection": 718,
                    "beginSection": "sections.0",
                    "endSection": "section.0"
                }, ...
            ],
            "concepts": [
                "http://www.diseaseontology.org/api/metadata/DOID:7148"
                , ...
            ],
            "triples": [
                {
                    "s": "http://linkedlifedata.com/resource/umls/id/C2827401",
                    "p": "http://www.w3.org/2008/05/skos-xl#prefLabel",
                    "o": "http://linkedlifedata.com/resource/umls/label/A17680439"
                },...
            ]
        }, ...
    ]}
    """
    concepts = dict()
    # Concepts need to be grouped by qids
    for c in results['concepts']:
        if c['id'] not in concepts:
            concepts[c['id']] = list()
        concepts[c['id']].append(c)
    articles = results['articles']  # RankdedDocs
    rdfs = results['RDF']

    output = dict()
    output['questions'] = []
    res_keys = ['type', 'body', 'id', 'documents', 'snippets', 'concepts',
                'triples']
    for q in questions:
        entry = dict.fromkeys(res_keys)
        # Question id
        entry['id'] = q['id']
        entry['type'] = q['type']
        entry['body'] = q['body']
        # Concepts
        MeSH, GO, UniProt, Jochem, DO = (set() for _ in range(5))
        entry['concepts'] = []
        if q['id'] in concepts:
            for c in concepts[q['id']]:
                if c['source'] == 'MetaMap (MeSH)':
                    MeSH.add(c['cid'])
                if c['source'].startswith('TmTools') and 'MESH' in c['cid']:
                    MeSH.add(c['cid'].split(':')[-1])
                if c['source'] == 'GO':
                    GO.add(c['cid'])
                if c['source'] == 'UniProt':
                    UniProt.add(c['cid'])

            # MeSH
            tmpl_ = "https://meshb.nlm.nih.gov/record/ui?ui={}"
            entry['concepts'].extend([tmpl_.format(id) for id in MeSH])
            # GO
            tmpl_ = "http://amigo.geneontology.org/cgi-bin/amigo/term_details" \
                    "?term={}"  # 2017
            entry['concepts'].extend([tmpl_.format(id) for id in GO])
            # Uniprot
            tmpl_ = "http://www.uniprot.org/uniprot/{}"
            entry['concepts'].extend([tmpl_.format(id) for id in UniProt])
            entry['concepts'] = entry['concepts'][:10]
        # Articles
        if q['id'] not in articles:
            entry['documents'] = []
        else:
            scores = articles[q['id']].rankings_fusion[:10]
            tmpl_ = "http://www.ncbi.nlm.nih.gov/pubmed/{}"
            entry['documents'] = [tmpl_.format(s) for s in scores]
        # Text Snippets (that appear in the returned article list)
        snp_appeared = []
        if q['id'] in articles and len(articles[q['id']].text_snippets) > 0:
            snippets_ = articles[q['id']].text_snippets
            for s in snippets_:
                if s[0]['document'] in entry['documents']:
                    if len(s[0]['text'].split()) >= 3:  # Don't want KEYWORDS
                        snp_appeared.append(s[0])
            entry['snippets'] = snp_appeared[:10]
        else:
            entry['snippets'] = []

        # RDFs
        if q['id'] not in rdfs:
            entry['triples'] = []
        else:
            entry['triples'] = []
            pass
        output['questions'].append(entry)

    # Write out
    filename = os.path.basename(args.dryrun_file).split('.')[0]
    output_file = os.path.join(PATHS['runs_dir'], filename+'_submit.json')

    logger.info('Writing the results on {}'.format(output_file))
    json.dump(output, open(output_file, 'w'), indent=4, separators=(',', ': '))


def test():
    batch_report = '== Results over batches ==\n'
    for i, b in enumerate(map(int, args.batch.split(','))):
        lstRankedDocs = []
        stats = {
            'avg_prec': AverageMeter(),
            'avg_recall': AverageMeter(),
            'avg_f1': AverageMeter(),
            'map': AverageMeter(),
            'logp': AverageMeter()
        }
        for seq, q in enumerate(questions[i]):
            logger.info('Querying: {}'.format(q['id']))
            res = query(q)
            topn = args.ndocs if args.qid is not None or args.verbose else 10
            res.write_result(printout=args.verbose, stats=stats, topn=topn,
                             seq=(seq, len(questions[i])))
            lstRankedDocs.append(res)
        # Report the overall batch performance measures
        gmap = math.exp(stats['logp'].avg)
        report = ("[Test Run #{} batch #{}] "
                  "prec.: {:.4f}, recall: {:.4f}, f1: {:.4f} "
                  "map: {:.4f}, gmap: {:.4f}"
                  ).format(args.run_id, b, stats['avg_prec'].avg,
                           stats['avg_recall'].avg, stats['avg_f1'].avg,
                           stats['map'].avg, gmap)
        logger.info(report)
        batch_report += report + '\n'
        if args.output_scores:
            save_results_test(lstRankedDocs, year=args.year, batch=b)
    if args.verbose:
        print(batch_report)
    else:
        logger.info(batch_report)


def dryrun():
    results = dict()
    # Retrieve concepts
    cr = ConceptRetriever(updateDatabase=args.update_concepts)
    results['concepts'] = cr.get_concepts(questions)
    # Retrieve articles and snippets
    results['articles'] = dict()
    for seq, q in enumerate(questions):
        res = query(q)
        add_results(res, articles=results['articles'],
                    seq=(seq, len(questions)))
    # RDF triples
    results['RDF'] = dict()
    save_results_dry(questions, results)


if __name__ == '__main__':
    # Global
    logger = logging.getLogger()
    # Set Options
    parser = argparse.ArgumentParser(
        'BioAsq6B', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    # Initialize
    (questions, doc_ranker, qasim_ranker, journal_ranker, semmeddb_ranker) = \
        init()
    unseen_words = set()

    # # RUN; For training, use scripts/data_entities/interactive_train.py
    if args.mode == 'test':
        test()
    elif args.mode == 'dry':
        dryrun()
