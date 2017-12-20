#!/usr/bin/env python3
"""This script converts PubMed Article document (medline) into TREC text
format in order to be used by Galago build index process."""

import argparse
from lxml import etree
import os
import logging
from tqdm import tqdm
import re
import gzip
from multiprocessing import Pool, cpu_count
from collections import OrderedDict


def convert(file):

    # get the basename of the path, and unique filename
    filename = os.path.basename(file).split('.')[0]
    output_path = os.path.join(args.output_path, filename)

    # create output directory
    os.makedirs(output_path, exist_ok=True)

    if file.endswith('gz'):
        root = etree.parse(gzip.open(file))
    else:
        root = etree.parse(open(file))

    batch_seq = 0
    doc_seq = 0
    doc_content = []
    for doc in root.findall('PubmedArticle'):
        data = OrderedDict.fromkeys(
            ["DOCNO", "TEXT", "TITLE", "JOURNAL_TITLE", "ISO_ABRV",
             "MEDLINE_TA", "CHEMICAL_UI", "CHEMICAL",
             "MESH_UI", "MESH_DESC", "MESH_QUAL", "KEYWORDS"])
        data['DOCNO'] = 'PMID-' + doc.findtext('.//PMID')
        data['TEXT'] = doc.findtext('.//Abstract/AbstractText')
        data['JOURNAL_TITLE'] = doc.findtext('.//Journal/Title')
        data['ISO_ABRV'] = doc.findtext('.//Journal/ISOAbbreviation')
        data['MEDLINE_TA'] = doc.findtext('.//MedlineTA')
        data['TITLE'] = doc.findtext('.//ArticleTitle')
        # text field required
        if data['TEXT'] is None:
            data['TEXT'] = ' '
        chemical_ui = []
        chemical = []
        for chem in doc.findall('.//Chemical'):
            subs = chem.find('NameOfSubstance')
            if subs is not None:
                chemical_ui.append(subs.attrib.get('UI'))
                chemical.append(subs.text)
        if len(chemical_ui) > 0:
            data['CHEMICAL_UI'] = chemical_ui
        if len(chemical) > 0:
            data['CHEMICAL'] = chemical
        mesh_ui = []
        mesh_descriptor = []
        mesh_qualifier = []
        for mesh in doc.findall('.//MeshHeading'):
            descriptor = mesh.find('DescriptorName')
            if descriptor is not None:
                mesh_ui.append(descriptor.attrib.get('UI'))
                mesh_descriptor.append(descriptor.text)
            qualifier = mesh.find('QualifierName')
            if qualifier is not None:
                mesh_ui.append(qualifier.attrib.get('UI'))
                mesh_qualifier.append(qualifier.text)
        if len(mesh_ui) > 0:
            data['MESH_UI'] = mesh_ui
        if len(mesh_descriptor) > 0:
            data['MESH_DESC'] = mesh_descriptor
        if len(mesh_qualifier) > 0:
            data['MESH_QUAL'] = mesh_qualifier
        keywords = []
        for keyword in doc.findall('.//Keyword'):
            keywords.append(keyword.text)
        if len(keywords) > 0:
            data['KEYWORDS'] = keywords

        doc_seq += 1
        doc_lines = ['<DOC>']
        for k, v in data.items():
            if v:
                doc_lines.append('<{}>{}</{}>'.format(k, v, k))
        doc_lines.append('</DOC>')
        doc_content.append('\n'.join(doc_lines))
        if doc_seq % 1000 == 0:
            filename = "batch{:02d}.dat".format(batch_seq)
            with open(os.path.join(output_path, filename), 'w') as out_f:
                out_f.write('\n\n'.join(doc_content))
            doc_content = []
            batch_seq += 1
    # write out the rest
    if len(doc_content) > 0:
        filename = "batch{:02d}.dat".format(batch_seq)
        with open(os.path.join(output_path, filename), 'w') as out_f:
            out_f.write('\n\n'.join(doc_content))

    return file


def run():
    assert os.path.exists(args.input_path), "input_path doesn't exist"
    assert os.path.exists(args.output_path), "output_path doesn't exist"

    # read all the paths to the input documents
    doc_files = []
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if not file.endswith('gz') and not file.endswith('xml'):
                continue
            doc_files.append(os.path.join(root, file))
    logger.info('{} medline files found from {}'
                ''.format(len(doc_files), args.input_path))

    logger.info('converting...')
    pool = Pool(processes=args.num_workers)
    for f in tqdm(pool.imap_unordered(convert, doc_files),
                  total=len(doc_files)):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='/path/to/input_docs')
    parser.add_argument('output_path', type=str, help='/path/to/output_docs')
    parser.add_argument('--num_workers', type=int, default=cpu_count(),
                        help='number of processes for converting')
    args = parser.parse_args()

    # initialize logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    run()
