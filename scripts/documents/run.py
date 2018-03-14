#!/usr/bin/env python3
"""User script that can be run for both optimization and evaluation on BioASQ
QA datasets"""

import argparse
import os
import sys
import json
import prettytable
import random
import re
import math
import multiprocessing
from multiprocessing import Pool, Manager
from datetime import datetime
from termcolor import colored
import logging
import pickle
from functools import partial
from collections import OrderedDict
import traceback   # may need to use this to get the traceback generated
                     # from inside a thread or process

from BioAsq6B import retriever, reranker, PATHS
from BioAsq6B.common \
    import AverageMeter, measure_performance, AdaptiveRandomSearch
from BioAsq6B.data_services import ConceptRetriever


def init():
    """Set default values and initialize components"""
    global doc_ranker, re_ranker

    # --------------------------------------------------------------------------
    # Set default options
    # --------------------------------------------------------------------------
    if args.run_id is None:
        args.run_id = datetime.today().strftime("%b%d-%H%M")
    else:
        args.run_id = datetime.today().strftime("%b%d-%H%M-") + args.run_id
    PATHS['log_file'] = os.path.join(PATHS['runs_dir'], args.run_id + '.log')
    if args.qid:
        if args.mode == 'train':
            logger.warning("Changing the run mode to 'test' to examine on one "
                           "QA pair")
            args.mode = 'test'
        args.year = 6  # By so, force to read all pairs as testing dataset
    if args.cache_scores:
        score_datafile = 'qa_prox/var/qa_scores-{}.pkl'.format(args.run_id)
        args.score_datafile = os.path.join(PATHS['data_dir'], score_datafile)

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
    # Doc-ranker
    logger.info('initializing retriever...')
    if args.mode == 'train' and args.cache_retrieval:
        logger.info('cache_retrieval created')
        manager = Manager()
        cached_retrievals = manager.dict()
        doc_ranker = \
            retriever.get_class('galago')(args,
                                          cached_retrievals=cached_retrievals)
    else:
        doc_ranker = retriever.get_class('galago')(args)
    # Re-ranker
    if args.rerank:
        logger.info('initializing re-ranker...')
        # If using the cached QA_sim scores for faster evaluation
        cached_scores = None
        if args.cache_scores:
            manager = Manager()
            if os.path.isfile(args.score_datafile):
                # Confirm if a user wants to read the cached scores, or start
                # from the scratch
                ans = input("Cached proximity scores exist. Do you want to read"
                            " [Y/n]? ")
                if ans.lower().startswith('n'):
                    cached_scores = manager.dict()
                else:
                    print('Reading the scores: {}'.format(args.score_datafile))
                    with open(args.score_datafile, 'rb') as f:
                        scores = pickle.load(f)
                    cached_scores = manager.dict(scores)
            else:
                cached_scores = manager.dict()

        re_ranker = reranker.RerankQaProx(args, cached_scores=cached_scores)
        if args.print_parameters:
            from BioAsq6B.qa_proximity import utils
            model_summary = utils.torch_summarize(re_ranker.predictor.model)
            logger.info(model_summary)
        # Check if the model needs to load idf data
        if re_ranker.predictor.model.args.use_idf:
            try:
                idf = pickle.load(open(PATHS['idf_file'], 'rb'))
            except:
                logger.error('Failed to read idf file from {}'
                             ''.format(PATHS['idf_file']))
                raise
            logger.info('Using idf feature: {} loaded'.format(len(idf)))
            re_ranker.predictor.idf = idf


def add_arguments(parser):
    """Define parameters with user provided arguments"""
    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1', 'y')

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
    runtime.add_argument('--debug', action='store_true')
    # Retriever Settings
    retriever = parser.add_argument_group('Retriever Settings')
    retriever.add_argument('--galago-weights', type=str, default=None,
                           help='The weights used in galago query statements')
    retriever.add_argument('--ndocs', type=int, default=10,
                           help='Number of document to retrieve')
    retriever.add_argument('-r', '--cache-retrieval', action='store_true',
                           help='Enable caching galago retrieval results')
    retriever.add_argument('-c', '--cache-scores', action='store_true',
                           help='Enable caching of the qa_sim scores')
    # Reranker settings
    reranker = parser.add_argument_group('Reranker Settings')
    reranker.add_argument('--rerank', action='store_true',
                           help='Enable re-ranker using qa_proximity model')
    reranker.add_argument('--score-weights', type=str, default=None,
                           help='comma separated weights of scoring functions')
    reranker.add_argument('--score-fusion', type=str, default='weighted_sum',
                           choices=['weighted_sum', 'rrf'],
                           help='Score fusion method')
    reranker.add_argument('--query-model', type=str, default='sdm',
                           help='document retrieval model')
    # Model Architecture: model specific options
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--qasim-model', type=str,
                        help='Path to a QA_Similarity model')
    model.add_argument('--print-parameters', action='store_true',
                       help='Print out model parameters')


def write_result_articles(res, stats=None):
    """Write the results with performance measures"""
    # Set tabular format for eval results
    table = prettytable.PrettyTable(['Question', 'GT', 'Returned', 'Scores'])
    table.align['Question'] = 'l'
    table.align['Returned'] = 'r'
    table.max_width['Question'] = 40
    col0 = '[{}]\n{}'.format(res['question'][0], res['question'][1])
    if args.qid and 'ideal_answer' in res:
        col0 += '\n=> {}'.format(res['ideal_answer'])
    col1 = '\n'.join(res['d_exp'])
    col2 = []  # returned documents
    col3 = []  # scores
    for k, v in res['scores'].items():
        k = colored(k, 'blue') if k in res['d_exp'] else k
        col2.append('{:>8}'.format(k))
        if 'rel_scores' in v:
            col3.append('({:.2f}, ({:.2f}/{:.2f}/{:.2f}), {:.4E})'
                        ''.format(v['ret_score'], v['rel_scores'][0],
                                  v['rel_scores'][1], v['rel_scores'][2],
                                  v['score']))
        else:
            col3.append('({:.4f}, {:.4E})'
                        ''.format(v['ret_score'], v['score']))
    col2 = '\n'.join(col2)
    col3 = '\n'.join(col3)
    table.add_row([col0, col1, col2, col3])
    prec, recall, f1, ap = \
        measure_performance(res['d_exp'], list(res['scores']), cutoff=10)
    report = ('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
              'avg_precision: {:.4f}').format(prec, recall, f1, ap)
    # Write out
    if args.verbose:
        logger.info("Details:\n" + table.get_string() + '\n')
    logger.info('[#{} {}/{}] {}'.format(res['question'][0], res['seq'][0],
                                         res['seq'][1], report))
    # Update statistics
    if stats:
        stats['avg_prec'].update(prec)
        stats['avg_recall'].update(recall)
        stats['avg_f1'].update(f1)
        stats['map'].update(ap)
        stats['logp'].update(math.log(prec + 1e-6))
    return


def write_result_articles_dry(res):
    """Different version of write_result function for dry-run; No performance
    measure applied"""
    # Set tabular format for eval results
    table = prettytable.PrettyTable(['Question', 'Returned', 'Scores'])
    table.align['Question'] = 'l'
    table.align['Returned'] = 'r'
    table.max_width['Question'] = 40
    col0 = '[{}]\n{}'.format(res['question'][0], res['question'][1])
    col1 = []  # returned documents
    col2 = []  # scores
    for k, v in res['scores'].items():
        col1.append('{:>8}'.format(k))
        if 'rel_scores' in v:
            col2.append('({:.2f}, ({:.2f}/{:.2f}/{:.2f}), {:.4E})'
                        ''.format(v['ret_score'], v['rel_scores'][0],
                                  v['rel_scores'][1], v['rel_scores'][2],
                                  v['score']))
        else:
            col2.append('({:.4f}, {:.4E})'
                        ''.format(v['ret_score'], v['score']))
    col1 = '\n'.join(col1)
    col2 = '\n'.join(col2)
    table.add_row([col0, col1, col2])
    # Write out
    if args.verbose:
        logger.info("Details:\n" + table.get_string() + '\n')
        # print(res['snippets'][:10])
    return


def query(q, seq=None):
    """run retrieval procedure (optionally rerank) of one question and return
    the result"""
    results = dict()
    results['question'] = [q['id'], q['body']]
    if 'ideal_answer' in q:
        results['ideal_answer'] = q['ideal_answer']
    if seq:
        results['seq'] = seq
    else:
        results['seq'] = (1, 1)
    (docids, ret_scores) = doc_ranker.closest_docs(q, k=args.ndocs)
    if args.rerank:
        (rel_scores, snippets) = re_ranker.get_prox_scores(docids, q)
        results['scores'] = re_ranker.merge_scores(args.score_weights, docids,
                                                   ret_scores, rel_scores)
        results['snippets'] = snippets
    else:
        _scores = {docid: {'ret_score': score, 'score': score}
                   for docid, score in zip(docids, ret_scores)}
        results['scores'] = OrderedDict(sorted(_scores.items(),
                                               key=lambda t: t[1]['score'],
                                               reverse=True))
    # Read expected documents
    if args.mode != 'dry':
        results['d_exp'] = []
        for line in q['documents']:
            m = re.match(".*pubmed/(\d+)$", line)
            if m:
                results['d_exp'].append(m.group(1))
    return results


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
            write_result_articles(res, stats)
        else:
            # Callback function to write the results of queries
            cb_write_results = partial(write_result_articles, stats=stats)
            # Generate pool for multiprocessing
            p = Pool(16)
            count = 0
            for q in test_data[b]:
                p.apply_async(query, args=(q, (count, len(test_data[b]))),
                              callback=cb_write_results)
                count += 1
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
    """Objective function for optimization; Simply returns cost of each epoch"""
    args.score_weights = ','.join(['{:.4f}'.format(w) for w in vec])
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
    # Callback function to write the results of queries
    cb_write_results = partial(write_result_articles, stats=stats)
    # Generate Pool for multiprocessing
    p = Pool(16)
    for seq, q in enumerate(samples):
        p.apply_async(query, args=(q, (seq, len(samples))),
                      callback=cb_write_results)
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
    # if caching score is enabled, store the scores
    if args.cache_scores:
        logger.info("Proximity Scores are saved")
        pickle.dump(dict(re_ranker.cached_scores),
                    open(args.score_datafile, 'wb'))
    # Returns (1 - 'map' score) as the cost
    return (1 - stats['map'].avg)


def train1():
    """Hyperparameter optimization: adaptive random search"""
    """Best Weights: [0.8554,0.1228,0.0210,0.1825]"""
    initial_weights = list(map(float, args.score_weights.split(',')))
    optimizer = AdaptiveRandomSearch([[0, 1]] * 4, objective_fn,
                                     initial_weights, max_iter=500)
    best = optimizer.search()
    logger.info("BEST params: {}".format(best))


def train2():
    """Optimize the weights used in galago queries (text, mesh_desc, mesh_ui)"""
    initial_weights = list(map(float, args.galago_weights.split(',')))
    optimizer = AdaptiveRandomSearch([[0, 5]] * 2, objective_fn2,
                                     initial_weights, max_iter=500)
    best = optimizer.search()
    logger.info("BEST params: {}".format(best))


def save_results_dry(questions, concepts, articles, rdfs):
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
            # tmpl_ = "https://www.nlm.nih.gov/cgi/mesh/2017/MB_cgi?field=uid" \
            #         "&exact=Find+Exact+Term&term={}"  # 2017
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
            scores = list(articles[q['id']]['scores'].items())[:10]
            tmpl_ = "http://www.ncbi.nlm.nih.gov/pubmed/{}"
            entry['documents'] = [tmpl_.format(s[0]) for s in scores]
        # Text Snippets (that appear in the returned article list)
        snp_appeared = []
        if 'snippets' in articles[q['id']]:
            for s in articles[q['id']]['snippets']:
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


def dryrun():
    assert args.dryrun_file is not None
    # Read dryrun file
    with open(args.dryrun_file) as f:
        data = json.load(f)
        questions = data['questions'] \
            if not args.debug else data['questions'][:3]
    logger.info('{} quetions read'.format(len(questions)))
    # Retrieve concepts
    cr = ConceptRetriever(args, updateDatabase=args.update_concepts)
    concepts = cr.get_concepts(questions)
    # Retrieve articles and snippets
    articles = dict()
    def add_results(res):
        articles[res['question'][0]] = res
        write_result_articles_dry(res)
    p = Pool(16)
    for q in questions:
        p.apply_async(query, args=(q,), callback=add_results)
    p.close()
    p.join()
    # RDF triples
    RDFs = dict()
    save_results_dry(questions, concepts, articles, RDFs)


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
