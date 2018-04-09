#!/usr/bin/env python3
"""User script that can be run for both optimization and evaluation on BioASQ
QA datasets"""

import argparse
import os
import sys
import json
import random
import math
import multiprocessing
from multiprocessing import Pool, Manager
from datetime import datetime
import logging
import pickle
from functools import partial
import traceback   # may need to use this to get the traceback generated
                     # from inside a thread or process

from BioAsq6B.retriever import GalagoSearch
from BioAsq6B import reranker, PATHS, Cache
from BioAsq6B.common \
    import AverageMeter, measure_performance, AdaptiveRandomSearch
from BioAsq6B.data_services import ConceptRetriever


def init():
    """Set default values and initialize components"""
    global doc_ranker, qasim_ranker, journal_ranker, cache
    doc_ranker = qasim_ranker = journal_ranker = None
    # --------------------------------------------------------------------------
    # Set default options
    # --------------------------------------------------------------------------
    if args.score_weights is not None:
        args.rerank = []
        options = ['qasim', 'journal']
        for scheme in args.score_weights.split(','):
            key, weight = scheme.split(':')
            if key in options:
                args.rerank.append(key)

    if args.run_id is None:
        args.run_id = datetime.today().strftime("%b%d-%H%M")
    PATHS['log_file'] = os.path.join(PATHS['runs_dir'], args.run_id + '.log')
    if args.qid and args.mode == 'train':
        logger.warning("Changing the run mode to 'test' to examine on one "
                       "QA pair")
        args.mode = 'test'
        args.year = 6  # By so, force to read all pairs as testing dataset

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
    if args.mode == 'train':
        if args.year >= 5:  # Read from 'BioASQ-trainingDataset'
            # Read training datasets
            batch_file = 'BioASQ-trainingDataset{}b.json'.format(args.year)
            logger.info('Reading train dataset: {}'.format(batch_file))
            batch_file = os.path.join(PATHS['train_dir'], batch_file)
            with open(batch_file) as f:
                data = json.load(f)
                train_data.extend(data['questions'])
        else:  # Read from previous 'test' datasets and use as training
            for y in range(2, args.year):
                for b in range(1, 6):
                    batch_file = 'phaseB_{}b_0{}.json'.format(y, b)
                    logger.info('Reading train dataset: {}'.format(batch_file))
                    batch_file = os.path.join(PATHS['test_dir'], batch_file)
                    with open(batch_file) as f:
                        data = json.load(f)
                        train_data.extend(data['questions'])
    # Reading test datasets.
    if args.mode == 'test' and args.year == 6:
        if args.qid:  # In order to find the question from all examples
            batch_file = 'BioASQ-trainingDataset{}b.json'.format(args.year)
            logger.info('Reading train dataset: {}'.format(batch_file))
            batch_file = os.path.join(PATHS['train_dir'], batch_file)
            with open(batch_file) as f:
                data = json.load(f)
                for q in data['questions']:
                    if args.qid == q['id']:
                        test_data.append([q])
        else:
            raise RuntimeError('No test data exist for the year 6')

    if args.mode != 'dry' and args.year != 6:
        batches = list(range(1, 6)) if args.batch is None else [args.batch]
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
                    test_data.append(data['questions'])
    if len(train_data) + len(test_data) > 0:
        logger.info('Training pairs: {}, Testing pairs: {}'
                    ''.format(len(train_data), [len(b) for b in test_data]))
    # --------------------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------------------
    cache = Cache(args)
    logger.info('initializing retriever...')
    doc_ranker = GalagoSearch(args)

    # QaSim ranker
    if 'qasim' in args.rerank:
        logger.info('initializing QaSim-ranker...')
        qasim_ranker = reranker.RerankQaSim(args)
        if args.print_parameters:
            from BioAsq6B.qa_proximity import utils
            model_summary = utils.torch_summarize(qasim_ranker.predictor.model)
            logger.info(model_summary)
        # Check if the model needs to load idf data
        if 'idf' in qasim_ranker.predictor.model.conf['features']:
            try:
                idf = pickle.load(open(PATHS['idf_file'], 'rb'))
            except:
                logger.error('Failed to read idf file from {}'
                             ''.format(PATHS['idf_file']))
                raise
            logger.info('Using idf feature: {} loaded'.format(len(idf)))
            qasim_ranker.predictor.idf = idf

    # Journal ranker
    if 'journal' in args.rerank:
        logger.info('initializing Journal-ranker...')
        journal_ranker = reranker.RerankerJournal()


def add_arguments(parser):
    """Define parameters with user provided arguments"""
    # Runtime Settings
    runtime = parser.add_argument_group('Runtime Settings')
    runtime.add_argument('--mode', type=str, default='train',
                         choices=['test', 'train', 'dry'],
                         help='Run mode; either test or train')
    runtime.add_argument('--run-id', type=str, default=None,
                         help='Identifiable name for each run')
    runtime.add_argument('-y', '--year', type=int, default=6,
                         choices=[3, 4, 5, 6],
                         help='Specify the year of the dataset with which an '
                              'evaluation will be done')
    runtime.add_argument('-b', '--batch', type=int,
                         help='Specify the batch number to be tested')
    runtime.add_argument('-q', '--qid', type=str, default=None,
                        help="One question ID to evaluate on")
    runtime.add_argument('-s', '--sample-size', type=float, default=.2,
                         help='Sample of BioAsq training dataset;'
                              '< 1 percentage, >= 1 num. of samples, 0 all')
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
    retriever.add_argument('-c', '--use-cache-scores', action='store_true',
                           help='Use cached retrieval and QaSim scores over '
                                'qids')
    # Reranker settings
    reranker = parser.add_argument_group('Reranker Settings')
    reranker.add_argument('--rerank', type=str, default='',
                          help='Enable specified rerankers in comma separated '
                               'format; choices ["qasim", "journal"]')
    reranker.add_argument('--score-weights', type=str, default="retrieval:1",
                          help='comma separated weights for score fusion; ex. '
                               '"retrieval:0.7,qasim:0.15,journal:0.15"')
    reranker.add_argument('--score-fusion', type=str, default='weighted_sum',
                          choices=['weighted_sum', 'rrf'],
                          help='Score fusion method')
    reranker.add_argument('--query-model', type=str, default='sdm',
                          help='document retrieval model')
    reranker.add_argument('--word-dict-file', type=str, default='word_dict.pkl',
                          help='Path to word_dict file for test/dry run')
    # Model Architecture: model specific options
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--qasim-model', type=str,
                        help='Path to a QA_Similarity model')
    model.add_argument('--print-parameters', action='store_true',
                       help='Print out model parameters')


def write_result_articles(res, stats=None, seq=None):
    """Write the results with performance measures"""
    if seq is not None:
        logger.info('=== {} / {} ==='.format(*seq))
    res.write_result(printout=args.verbose, stats=stats)
    cache.update_scores(res)  # Update scores if necessary
    return


def add_results(res, articles=None, seq=None):
    if seq is not None:
        logger.info('=== {} / {} ==='.format(*seq))
    articles[res.query['id']] = res
    res.write_result(printout=args.verbose)
    cache.update_scores(res)  # Update scores if necessary
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
        # todo. After qasim, you can build a list of text snippets
        # ranked_docs.build_text_snippets()
    if 'journal' in args.rerank:
        if cache.flg_update_scores['journal']:
            journal_ranker.get_journal_scores(ranked_docs)
        else:
            journal_ranker.get_journal_scores(ranked_docs, cache=cache_score)
    ranked_docs.merge_scores(args.score_weights)
    return ranked_docs


def sample_questions():
    assert args.mode == 'train'
    if args.sample_size < 1:  # Sample size is in percentage
        sample_size = int(len(train_data) * args.sample_size)
        if args.sample_size == 0:
            sample_size = len(train_data)
    else:  # Sample size in the number of examples
        sample_size = int(args.sample_size)

    # Sampling
    if args.random_seed:
        if args.random_seed == 0:
            random.seed()
        else:
            random.seed(args.random_seed)
    if sample_size == len(train_data):
        samples = train_data
    else:
        samples = random.sample(train_data, sample_size)
    logger.info('# of questions: {}, sample size: {}'
                ''.format(len(train_data), sample_size))
    return samples


def objective_fn(vec):
    """For optimization; Simply returns the cost of each epoch"""
    args.score_weights = 'retrieval:{:.4f},journal:{:.4f}'.format(*vec)
    return _run_epoch()


def objective_fn2(vec):
    """Objective function for optimization; Simply returns cost of each epoch"""
    args.galago_weights = ','.join(['{:.4f}'.format(w) for w in vec])
    return _run_epoch()


def _run_epoch():
    samples = sample_questions()
    stats = {
        'avg_prec': AverageMeter(),
        'avg_recall': AverageMeter(),
        'avg_f1': AverageMeter(),
        'map': AverageMeter(),
        'logp': AverageMeter()
    }
    # Generate Pool for multiprocessing
    p = Pool(16)
    for seq, q in enumerate(samples):
        # Callback function to write the results of queries
        cb_write_results = \
            partial(write_result_articles,
                    seq=(seq, len(samples)), stats=stats)
        p.apply_async(query, args=(q, ), callback=cb_write_results)
    p.close()
    p.join()
    # Report the overall batch performance measures
    gmap = math.exp(stats['logp'].avg)
    report = ("[Test Run #{}] "
              "prec.: {:.4f}, recall: {:.4f}, f1: {:.4f} "
              "map: {:.4f}, gmap: {:.4f}"
              ).format(args.run_id, stats['avg_prec'].avg,
                       stats['avg_recall'].avg, stats['avg_f1'].avg,
                       stats['map'].avg, gmap)
    if args.score_weights:
        logger.info('current weights: {}'.format(args.score_weights))
    if args.galago_weights:
        logger.info('current weights: {}'.format(args.galago_weights))
    logger.info(report)
    # Returns (1 - 'map' score) as the cost
    return (1 - stats['map'].avg)


def train1():
    """Hyperparameter optimization: adaptive random search"""
    """Best Weights: [0.8554,0.1228,0.0210,0.1825]"""
    """Best Weights: [0.74,0.15]"""
    assert args.score_weights is not None
    weights = []
    for w in args.score_weights.split(','):
        weights.append(float(w.split(':')[1]))
    # initial_weights = list(map(float, args.score_weights.split(',')))
    optimizer = AdaptiveRandomSearch([[0, 1]] * len(weights),
                                     objective_fn, weights, max_iter=100)
    best = optimizer.search()
    logger.info("BEST params: {}".format(best))


def train2():
    """Optimize the weights used in galago queries (text, mesh_desc, mesh_ui)"""
    initial_weights = list(map(float, args.galago_weights.split(',')))
    optimizer = AdaptiveRandomSearch([[0, 5]] * 2, objective_fn2,
                                     initial_weights,
                                     max_iter=100)
    best = optimizer.search()
    logger.info("BEST params: {}".format(best))


def save_results_dry(questions, results):
    concepts = results['concepts']
    articles = results['articles']
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
                if c['source'] == 'MetaMap':
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
        output['questions'].append(entry)
        # Articles
        if q['id'] not in articles:
            entry['documents'] = []
        else:
            scores = articles[q['id']].rankings_fusion[:10]
            # scores = list(articles[q['id']]['scores'].items())[:10]
            tmpl_ = "http://www.ncbi.nlm.nih.gov/pubmed/{}"
            entry['documents'] = [tmpl_.format(s) for s in scores]
        # Text Snippets (that appear in the returned article list)
        snp_appeared = []
        if q['id'] in articles and 'snippets' in articles[q['id']].docs_data:
            for s in articles[q['id']].docs_data['snippets']:
                if s[0]['document'] in entry['documents']:
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

    # Write out
    filename = os.path.basename(args.dryrun_file).split('.')[0]
    output_file = os.path.join(PATHS['runs_dir'], filename+'_submit.json')

    logger.info('Writing the results on {}'.format(output_file))
    json.dump(output, open(output_file, 'w'), indent=4, separators=(',', ': '))


def test():
    batch_report = '== Results over batches ==\n'
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
            p = Pool(16)
            for seq, q in enumerate(test_data[b]):
                # res = query(q)
                # write_result_articles(res, seq=(seq, len(test_data[b])),
                #                       stats=stats)
                # Callback function to write the results of queries
                cb_write_results = \
                    partial(write_result_articles,
                            seq=(seq, len(test_data[b])), stats=stats)
                p.apply_async(query, args=(q, ), callback=cb_write_results)
            p.close()
            p.join()
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


def dryrun():
    assert args.dryrun_file is not None
    results = dict()
    # Read dryrun file
    with open(args.dryrun_file) as f:
        data = json.load(f)
        questions = data['questions'] \
            if not args.debug else data['questions'][:2]
    if args.qid is not None:
        questions_ = []
        for q in questions:
            if q['id'] == args.qid:
                questions_.append(q)
        questions = questions_
    logger.info('{} questions read'.format(len(questions)))
    # Retrieve concepts
    cr = ConceptRetriever(updateDatabase=args.update_concepts)
    results['concepts'] = cr.get_concepts(questions)
    # Retrieve articles and snippets
    results['articles'] = dict()
    p = Pool(16)
    for seq, q in enumerate(questions):
        cb_add_results = partial(add_results, articles=results['articles'],
                                 seq=(seq, len(questions)))
        # res = query(q)
        # add_results(res, articles=results['articles'],
        #             seq=(seq, len(questions)))
        p.apply_async(query, args=(q, ), callback=cb_add_results)

    p.close()
    p.join()
    # RDF triples
    results['RDF'] = dict()
    save_results_dry(questions, results)


if __name__ == '__main__':
    # Global
    logger = logging.getLogger()
    test_data = []
    train_data = []
    # Set Options
    parser = argparse.ArgumentParser(
        'BioAsq6B', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    if args.word_dict_file:
        PATHS['word_dict_file'] = os.path.join(PATHS['data_dir'],
                                               args.word_dict_file)
    if args.qasim_model:
        # PATHS is defined in package __init__ file
        PATHS['qasim_model'] = \
            os.path.join(PATHS['data_dir'],
                         'qa_prox/var/{}'.format(args.qasim_model))
    # Initialize
    init()

    # RUN~
    if args.mode == 'test':
        test()
    elif args.mode == 'train':
        train1()
        # train2()
    elif args.mode == 'dry':
        dryrun()

    # Postprocess
    # Save cache data, if updated
    if cache.scores_changed:
        cache.save_scores()
    if cache.documents_cahnged:
        cache.save_docs()
    # - update word_dict (on dry run, new words can be found)
    if qasim_ranker is not None:
        qasim_ranker.predictor.update_word_dict()
