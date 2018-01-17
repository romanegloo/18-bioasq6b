#!/usr/bin/env python3
"""This script converts PubMed Article document (medline) into TREC text
format which to be used by Galago build index process."""

import argparse
from lxml import etree
import os
from tqdm import tqdm
import gzip
from multiprocessing import Pool, cpu_count
from functools import partial
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
    empty_count = 0
    doc_content = []
    for doc in root.findall('PubmedArticle'):
        data = OrderedDict.fromkeys(
            ["DOCNO", "TEXT", "TITLE", "JOURNAL_TITLE", "ISO_ABRV",
             "MEDLINE_TA", "CHEMICAL_UI", "CHEMICAL",
             "MESH_UI", "MESH_DESC", "MESH_QUAL", "KEYWORDS"])
        data['DOCNO'] = 'PMID-' + doc.findtext('.//PMID')
        data['TITLE'] = doc.findtext('.//ArticleTitle')
        abstract_text = []
        for t in doc.iterfind('.//Abstract/AbstractText'):
            if t.text is None:
                continue
            text_ = t.text
            if 'Label' in t.attrib:
                text_ = t.attrib['Label'] + ': ' + text_
            abstract_text.append(text_)
        data['TEXT'] = ' '.join(abstract_text)
        # If TEXT is empty, fill in the title.
        # (TEXT field is key field in trec format; Without it the entire
        # batch files will be ignored during indexing)
        if len(data['TEXT'].strip()) == 0:
            empty_count += 1
            data['TEXT'] = data['TITLE']
        data['JOURNAL_TITLE'] = doc.findtext('.//Journal/Title')
        data['ISO_ABRV'] = doc.findtext('.//Journal/ISOAbbreviation')
        data['MEDLINE_TA'] = doc.findtext('.//MedlineTA')
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

        doc_lines = ['<DOC>']
        for k, v in data.items():
            if v:
                doc_lines.append('<{}>{}</{}>'.format(k, v, k))
        doc_lines.append('</DOC>')
        doc_content.append('\n'.join(doc_lines))
        doc_seq += 1
        if mode == 'batch':
            if doc_seq % 1000 == 0:
                filename = "batch{:02d}.trectext".format(batch_seq)
                with open(os.path.join(output_path, filename), 'w') as out_f:
                    out_f.write('\n\n'.join(doc_content))
                doc_content = []
                batch_seq += 1
        else:
            filename = data['DOCNO'] + '.trectext'
            with open(os.path.join(output_path, filename), 'w') as out_f:
                out_f.write('\n'.join(doc_content))
            doc_content = []

    # write out the rest
    if mode == 'batch' and len(doc_content) > 0:
        filename = "batch{:02d}.trectext".format(batch_seq)
        with open(os.path.join(output_path, filename), 'w') as out_f:
            out_f.write('\n\n'.join(doc_content))
        pass

    return doc_seq, batch_seq, empty_count


def run():
    """create batches of trectext files"""
    assert os.path.exists(args.input_path), "input_path doesn't exist"
    assert os.path.exists(args.output_path), "output_path doesn't exist"

    # read all the paths to the input documents
    doc_files = []
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if not file.endswith('gz') and not file.endswith('xml'):
                continue
            doc_files.append(os.path.join(root, file))
    print('{} medline files found from {}'
          ''.format(len(doc_files), args.input_path))

    print('converting...')
    pool = Pool(processes=args.num_workers)
    total_doc = 0
    total_batch = 0
    total_empty = 0
    for d, b, n in tqdm(pool.imap_unordered(partial(convert), doc_files),
                        total=len(doc_files)):
        total_doc += d
        total_batch += b
        total_empty += n

    print('total docs: {}, total batches: {} created (empty doc {})'
          ''.format(total_doc, total_batch, total_empty))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='/path/to/input_docs')
    parser.add_argument('output_path', type=str, help='/path/to/output_docs')
    parser.add_argument('--num_workers', type=int, default=cpu_count(),
                        help='number of processes for converting')
    args = parser.parse_args()

    mode = 'batch'  # 'all' or 'batch'
    run()
