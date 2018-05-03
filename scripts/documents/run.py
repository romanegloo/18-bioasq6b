#!/usr/bin/env python3
"""User script that can be run for both optimization and evaluation on BioASQ
QA datasets"""

import argparse
import os
import sys
import json
import math
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import logging
import pickle
from functools import partial
import spacy
import traceback   # may need to use this to get the traceback generated
                     # from inside a thread or process

from BioAsq6B.retriever import GalagoSearch
from BioAsq6B import reranker, PATHS
from BioAsq6B.cache import Cache
from BioAsq6B.common import AverageMeter
from BioAsq6B.data_services import ConceptRetriever


def init():
    """Set default values and initialize components"""
    global doc_ranker, qasim_ranker, journal_ranker, semmeddb_ranker, cache, idf
    doc_ranker = qasim_ranker = journal_ranker = semmeddb_ranker = idf = None
    # --------------------------------------------------------------------------
    # Set default options
    # --------------------------------------------------------------------------
    if args.qasim_model is not None:
        # PATHS is defined in package __init__ file
        PATHS['qasim_model'] = os.path.join(PATHS['data_dir'],
                                      'qa_prox/var/{}'.format(args.qasim_model))
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
    logger.setLevel(logging.INFO)
    # also use this logger in the multiprocessing module
    mlp = multiprocessing.log_to_stderr()
    mlp.setLevel(logging.WARN)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    if args.verbose:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
    else:
        file = logging.FileHandler(PATHS['log_file'])
        file.setFormatter(fmt)
        logger.addHandler(file)
        print("writing output in {}".format(PATHS['log_file']))
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # --------------------------------------------------------------------------
    # Read datasets (question/answer pairs)
    # --------------------------------------------------------------------------
    # Reading test datasets.
    if args.mode != 'dry':
        if args.batch is None:
            batches = [1, 2, 3, 4, 5]
        else:
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
                            test_data.append([q])
                else:
                    if args.debug:
                        test_data.append(data['questions'][:3])
                    else:
                        test_data.append(data['questions'])
        if len(test_data) > 0:
            logger.info('Testing pairs: {}'.format([len(b) for b in test_data]))

    # --------------------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------------------
    cache = Cache(args)
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
            from BioAsq6B.qa_proximity import utils
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


def add_arguments(parser):
    """Define parameters with user provided arguments"""
    # Runtime Settings
    runtime = parser.add_argument_group('Runtime Settings')
    runtime.add_argument('--mode', type=str, default='test',
                         choices=['test', 'dry'],
                         help='Run mode; dry run results an output file')
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
    runtime.add_argument('--num-workers', type=int, default=40,
                         help='Number of workers in a multiprocessing pool')
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
                          choices=['baseline', 'sdm'],
                          help='document retrieval model')
    reranker.add_argument('--word-dict-file', type=str, default='word_dict.pkl',
                          help='Path to word_dict file for test/dry run')
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
    cache.update_scores(res)  # Update scores if necessary
    if len(res.unseen_words) > 0:
        global unseen_words
        unseen_words |= res.unseen_words
    return


def add_results(res, articles=None, seq=None):
    if seq is not None:
        logger.info('=== {} / {} ==='.format(*seq))
    articles[res.query['id']] = res
    res.write_result(printout=args.verbose)
    cache.update_scores(res)  # Update scores if necessary
    if len(res.unseen_words) > 0:
        global unseen_words
        unseen_words |= res.unseen_words
    return


def query(q):
    """run retrieval procedure (optionally rerank) of one question and return
    the result"""
    if q['id'] in cache.scores:
        cache_score = cache.scores[q['id']]
    else:
        cache_score = None
    if cache.flg_update_scores['retrieval']:
        ranked_docs = doc_ranker.closest_docs(q, k=args.ndocs)
    else:
        ranked_docs = doc_ranker.closest_docs(q, k=args.ndocs, cache=cache_score)
    # After retrieval, read documents
    cache_docs = {k: cache.documents[k] for k in ranked_docs.rankings
                  if k in cache.documents}
    ranked_docs.read_doc_text(cache=cache_docs)
    if 'qasim' in args.rerank:
        if cache.flg_update_scores['qasim']:
            qasim_ranker.get_qasim_scores(ranked_docs)
        else:
            qasim_ranker.get_qasim_scores(ranked_docs, cache=cache_score)
        ranked_docs.unseen_words = qasim_ranker.predictor.add_words
    if 'journal' in args.rerank:
        if cache.flg_update_scores['journal']:
            journal_ranker.get_journal_scores(ranked_docs)
        else:
            journal_ranker.get_journal_scores(ranked_docs, cache=cache_score)
    if 'semmeddb' in args.rerank:
        if cache.flg_update_scores['semmeddb']:
            semmeddb_ranker.get_semmeddb_scores(ranked_docs)
        else:
            semmeddb_ranker.get_semmeddb_scores(ranked_docs, cache=cache_score)

    ranked_docs.merge_scores(args.score_weights)
    return ranked_docs


def save_results_dry(questions, results):
    concepts = dict()
    # Concepts need to be grouped by qids
    for c in results['concepts']:
        if c['id'] not in concepts:
            concepts[c['id']] = list()
        concepts[c['id']].append(c)
    articles = results['articles']  # RankdedDocs
    rdfs = results['RDF']

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
    global unseen_words
    unseen_words = set()
    for b in range(len(test_data)):
        stats = {
            'avg_prec': AverageMeter(),
            'avg_recall': AverageMeter(),
            'avg_f1': AverageMeter(),
            'map': AverageMeter(),
            'logp': AverageMeter()
        }
        if args.qid:
            res = query(test_data[b][0])
            res.write_result(printout=True, stats=stats, topn=args.ndocs)
            cache.update_scores(res)  # Update scores if necessary
        else:
            # Generate pool for multiprocessing
            # p = Pool(16)
            for seq, q in enumerate(test_data[b]):
                res = query(q)
                write_result_articles(res, seq=(seq, len(test_data[b])),
                                      stats=stats)
                # Callback function to write the results of queries
                # cb_write_results = \
                #     partial(write_result_articles,
                #             seq=(seq, len(test_data[b])), stats=stats)
                # p.apply_async(query, args=(q, ), callback=cb_write_results)
            # p.close()
            # p.join()
        # Report the overall batch performance measures
        gmap = math.exp(stats['logp'].avg)
        report = ("[Test Run #{} batch #{}] "
                  "prec.: {:.4f}, recall: {:.4f}, f1: {:.4f} "
                  "map: {:.4f}, gmap: {:.4f}"
                  ).format(args.run_id, b+1, stats['avg_prec'].avg,
                           stats['avg_recall'].avg, stats['avg_f1'].avg,
                           stats['map'].avg, gmap)
        logger.info(report)
        batch_report += report + '\n'
    if args.verbose:
        print(batch_report)
    else:
        logger.info(batch_report)
    # Update word_dict (on dry run, new words can be found)
    if 'qasim' in cache.flg_update_scores and cache.flg_update_scores['qasim'] \
        and len(unseen_words) > 0:
        qasim_ranker.predictor.update_word_dict(unseen_words)


def dryrun():
    global unseen_words
    unseen_words = set()
    if args.dryrun_file is None:
        if args.year and args.batch:
            # Assuming a specific year and batch is provided
            args.dryrun_file = \
                os.path.join(PATHS['test_dir'],
                             'phaseA_{}b_0{}.json'.format(args.year,
                                                          args.batch))
        else:
            logger.error('Either dryrun-file or a specific year/batch needs '
                         'to be provided')
    results = dict()
    # Read dryrun file
    with open(args.dryrun_file) as f:
        data = json.load(f)
        questions = data['questions'] \
            if not args.debug else data['questions'][:3]
    logger.info('{} questions read'.format(len(questions)))
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
    # Update word_dict (on dry run, new words can be found)
    if 'qasim' in cache.flg_update_scores and cache.flg_update_scores['qasim'] \
            and len(unseen_words) > 0:
        qasim_ranker.predictor.update_word_dict(unseen_words)
    save_results_dry(questions, results)


if __name__ == '__main__':
    # Global
    logger = logging.getLogger()
    test_data = []
    # Set Options
    parser = argparse.ArgumentParser(
        'BioAsq6B', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    # Initialize
    init()

    # RUN; For training, use scripts/data_entities/interactive_train.py
    if args.mode == 'test':
        test()
    elif args.mode == 'dry':
        dryrun()

    # Postprocess; Save cache data, if updated
    if cache.scores_changed:
        cache.save_scores()
    if cache.documents_cahnged:
        cache.save_docs()

