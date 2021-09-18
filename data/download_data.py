#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to download various preprocessed data sources & checkpoints for DPR
"""

import gzip
import os
import pathlib

import argparse
import wget

NQ_LICENSE_FILES = [
    'https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE',
    'https://dl.fbaipublicfiles.com/dpr/nq_license/README',
]

RESOURCES_MAP = {
    'data.wikipedia_split.psgs_w100': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz',
        'original_ext': '.tsv',
        'compressed': True,
        'desc': 'Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)'
    },

    'compressed-data.wikipedia_split.psgs_w100': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz',
        'original_ext': '.tsv.gz',
        'compressed': False,
        'desc': 'Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)'
    },

    'data.retriever.nq-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'NQ dev subset with passages pools for the Retriever train time validation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.nq-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'NQ train subset with passages pools for the Retriever training',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.trivia-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'TriviaQA dev subset with passages pools for the Retriever train time validation'
    },

    'data.retriever.trivia-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'TriviaQA train subset with passages pools for the Retriever training'
    },

    'data.retriever.squad1-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'SQUAD 1.1 train subset with passages pools for the Retriever training'
    },

    'data.retriever.squad1-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'SQUAD 1.1 dev subset with passages pools for the Retriever train time validation'
    },

    'data.retriever.squad1-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'SQUAD 1.1 train subset with passages pools for the Retriever training'
    },

    'data.retriever.squad1-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'SQUAD 1.1 dev subset with passages pools for the Retriever train time validation'
    },

    'data.retriever.wq-train': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/Dq7EZpoHBj5FwKp/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'WebQuestions train subset with passages pools for the Retriever training'
    },

    'data.retriever.wq-dev': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/f4KE5aq8F8MaLLM/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'WebQuestions dev subset with passages pools for the Retriever train time validation'
    },

    'data.retriever.curated-train': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/EbyHeiRnCZqtoCe/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'CuratedTREC train subset with passages pools for the Retriever training'
    },

    'data.retriever.curated-dev': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/sTrbYGK3JPjmfxT/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'CuratedTREC dev subset with passages pools for the Retriever train time validation'
    },

    'data.retriever.qas.nq-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ dev subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ test subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ train subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },


    'data.retriever.qas.trivia-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'Trivia dev subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.trivia-test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'Trivia test subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.trivia-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'Trivia train subset for Retriever validation and IR results generation'
    },

    
    'data.retriever.qas.squad-dev': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/oQzj6zZzAyktCJ7/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'SQuAD 1.1 dev subset for Retriever validation and IR results generation'
    },
    
    'data.retriever.qas.wq-dev': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/wEkwRXXoMQjcZ5z/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'WebQuestions dev subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.wq-test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/cXczgxQZtJ4wY9d/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'WebQuestions test subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.wq-train': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/RHNXG6tWMBJRj8f/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'WebQuestions train subset for Retriever validation and IR results generation'
    },
    
    'data.retriever.qas.curated-dev': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/26nJrSeAYPwHxzn/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'CuratedTrec dev subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.curated-test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/tAnmyXXWKA4kdj8/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'CuratedTrec test subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.curated-train': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/ckGiSqm6gnk4gyE/download',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'CuratedTrec train subset for Retriever validation and IR results generation'
    },

    'data.gold_passages_info.nq_train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our train subset) gold positive passages and alternative question tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our dev subset) gold positive passages and alternative question tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our test, original dev subset) gold positive passages and alternative question '
                'tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'pretrained.fairseq.roberta-base.dict': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/dict.txt',
        'original_ext': '.txt',
        'compressed': False,
        'desc': 'Dictionary for pretrained fairseq roberta model'
    },

    'pretrained.fairseq.roberta-base.model': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/model.pt',
        'original_ext': '.pt',
        'compressed': False,
        'desc': 'Weights for pretrained fairseq roberta base model'
    },

    'pretrained.pytext.bert-base.model': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/pretrained/pytext/bert/bert-base-uncased.pt',
        'original_ext': '.pt',
        'compressed': False,
        'desc': 'Weights for pretrained pytext bert base model'
    },

    'data.retriever_results.nq.single.wikipedia_passages': {
        's3_url': ['https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq/wiki_passages_{}'.format(i) for i in
                   range(50)],
        'original_ext': '.pkl',
        'compressed': False,
        'desc': 'Encoded wikipedia files using a biencoder checkpoint('
                'checkpoint.retriever.single.nq.bert-base-encoder) trained on NQ dataset '
    },

    'data.retriever_results.nq.single.nq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/3Zobb6kX72twptY/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ test dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever_results.nq.single.trivia.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/7KAQGF653f5xRAQ/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of Trivia test dataset for the encoder trained on NQ',
    },
    
    'data.retriever_results.nq.single.curated.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/qZmLJnfCGXnwqFg/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of CuratedTrec test dataset for the encoder trained on NQ',
    },
    
    'data.retriever_results.nq.single.wq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/KbommFHDR7pznFF/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of WebQuestions test dataset for the encoder trained on NQ',
    },

    
    'data.retriever_results.trivia.single.nq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/dNzKkpW4Hqz2KmD/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ test dataset for the encoder trained on Trivia',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever_results.trivia.single.trivia.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/A2R3wbnJHaMQsZ2/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of Trivia test dataset for the encoder trained on Trivia',
    },
    
    'data.retriever_results.trivia.single.curated.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/2rLZagcRwebBymk/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of CuratedTrec test dataset for the encoder trained on Trivia',
    },
    
    'data.retriever_results.trivia.single.wq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/e3pDiyEqBjSJrWE/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of WebQuestions test dataset for the encoder trained on Trivia',
    },

    'data.retriever_results.curated.single.nq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/iSFm2mybbj5GTnA/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ test dataset for the encoder trained on Curated',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever_results.curated.single.trivia.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/4LwWAJpi24RDMXG/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of Trivia test dataset for the encoder trained on Curated',
    },
    
    'data.retriever_results.curated.single.curated.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/JtZoNSqAYtooH26/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of CuratedTrec test dataset for the encoder trained on Curated',
    },
    
    'data.retriever_results.curated.single.wq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/Y4WGQDtx4K5PcBq/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of WebQuestions test dataset for the encoder trained on Curated',
    },

    
    'data.retriever_results.wq.single.nq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/8tMm2KPfaemTmK7/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ test dataset for the encoder trained on WebQuestions',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever_results.wq.single.trivia.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/pqtN2xXb7nwDPkA/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of Trivia test dataset for the encoder trained on WebQuestions',
    },
    
    'data.retriever_results.wq.single.curated.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/nya77wtxMJtzyWr/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of CuratedTrec test dataset for the encoder trained on WebQuestions',
    },
    
    'data.retriever_results.wq.single.wq.test': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/24o3HDZ2nEZHJra/download',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of WebQuestions test dataset for the encoder trained on WebQuestions',
    },

    'checkpoint.retriever.single.nq.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/cE67piQ2enmDzGG/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on NQ data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.ensemble.nq.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/rQayq9nw5CxToDy/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Ensemble weights trained on NQ data and pre-trained HF bert-base-uncased model'
    },

    'checkpoint.retriever.single.trivia.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/ezTbxiMogkGPZs5/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on Trivia data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.ensemble.trivia.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/SFC8CW4X5AFz7HE/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Ensemble weights trained on Trivia data and pre-trained HF bert-base-uncased model'
    },

    'checkpoint.retriever.single.curated.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/yHT7MTcfzByaoQJ/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on Curated data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.ensemble.curated.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/4PfDrRprFHE7AQ8/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Ensemble weights trained on Curated data and pre-trained HF bert-base-uncased model'
    },

    'checkpoint.retriever.single.wq.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/cPt5Pf2jYk78ebc/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on WebQuestions data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.ensemble.wq.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/tjxRMB76P26s6Js/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Ensemble weights trained on WebQuestions data and pre-trained HF bert-base-uncased model'
    },

    'checkpoint.retriever.single.squad.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/NdB2X55zmd4rNmG/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on SQuAD data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.ensemble.squad.bert-base-encoder': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/Hjx5toCDzTq9q7F/download',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Ensemble weights trained on SQuAD data and pre-trained HF bert-base-uncased model'
    },

    # retrieval indexes
    'indexes.single.nq.flat.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/cQRp3HYSHWLBfPW/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on NQ-single retriever'
    },
    'indexes.single.nq.flat.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/YJgMyyLDQAGAGCM/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on NQ-single retriever (metadata)'
    },

    'indexes.single.trivia.flat.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/eGRpxXpD8KQbizT/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on Trivia-single retriever'
    },
    'indexes.single.trivia.flat.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/tgidtbpHLmWC4Bi/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on Trivia-single retriever (metadata)'
    },

    'indexes.single.squad.flat.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/RRfYDbzadZXAdkK/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on SQuAD-single retriever'
    },
    'indexes.single.squad.flat.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/tADiL6NS3b4EKPC/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on SQuAD-single retriever (metadata)'
    },

    'indexes.single.curated.flat.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/YDxwB6pQxnD2wHD/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on CuratedTrec-single retriever'
    },
    'indexes.single.curated.flat.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/kj8ZMdXqkb94RXX/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on CuratedTrec-single retriever (metadata)'
    },

    'indexes.single.wq.flat.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/g7TmjcC8y642osf/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on WebQuestions-single retriever'
    },
    'indexes.single.wq.flat.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/K6mX9ZRaYfAbWJK/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on WebQuestions-single retriever (metadata)'
    },

    # ivf indexes
    'indexes.single.nq.recon.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/r8zKZqeyDsejqPQ/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR flat index on NQ-single retriever'
    },
    'indexes.single.nq.recon.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/H78R2fFxJarmz4R/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on NQ-single retriever (metadata)'
    },

    'indexes.single.trivia.recon.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/esyafG6FLb9zKHN/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on Trivia-single retriever'
    },
    'indexes.single.trivia.recon.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/tnPyYWmp5Wqdqg9/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on Trivia-single retriever (metadata)'
    },

    'indexes.single.squad.recon.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/N8gXe9qWM6xyPL4/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on SQuAD-single retriever'
    },
    'indexes.single.squad.recon.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/SibmQgTWoAxyWMK/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on SQuAD-single retriever (metadata)'
    },

    'indexes.single.curated.recon.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/3AbEjYx6z7XM4pi/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on CuratedTrec-single retriever'
    },
    'indexes.single.curated.recon.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/NkoqC6SS4HjMGJJ/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on CuratedTrec-single retriever (metadata)'
    },

    'indexes.single.wq.recon.index': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/dAxD5t7tosXpMjt/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on WebQuestions-single retriever'
    },
    'indexes.single.wq.recon.index_meta': {
        's3_url': 'https://vault.cs.uwaterloo.ca/s/8Bxx3SbL4SfJxpQ/download',
        'original_ext': '.dpr',
        'compressed': False,
        'desc': 'DPR ivf index on WebQuestions-single retriever (metadata)'
    },
}


def unpack(gzip_file: str, out_file: str):
    print('Uncompressing ', gzip_file)
    input = gzip.GzipFile(gzip_file, 'rb')
    s = input.read()
    input.close()
    output = open(out_file, 'wb')
    output.write(s)
    output.close()
    print('Saved to ', out_file)


def download_resource(s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str) -> str:
    print('Loading from ', s3_url)

    # create local dir
    path_names = resource_key.split('.')

    root_dir = out_dir if out_dir else './'
    save_root = os.path.join(root_dir, *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file = os.path.join(save_root, path_names[-1] + ('.tmp' if compressed else original_ext))

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return save_root

    wget.download(s3_url, out=local_file)

    print('Saved to ', local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file)
        os.remove(local_file)
    return save_root


def download_file(s3_url: str, out_dir: str, file_name: str):
    print('Loading from ', s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return

    wget.download(s3_url, out=local_file)
    print('Saved to ', local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print('no resources found for specified key')
        return
    download_info = RESOURCES_MAP[resource_key]

    s3_url = download_info['s3_url']

    save_root_dir = None
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir = download_resource(url,
                                              download_info['original_ext'],
                                              download_info['compressed'],
                                              '{}_{}'.format(resource_key, i),
                                              out_dir)
    else:
        save_root_dir = download_resource(s3_url,
                                          download_info['original_ext'],
                                          download_info['compressed'],
                                          resource_key,
                                          out_dir)

    license_files = download_info.get('license_files', None)
    if not license_files:
        return

    download_file(license_files[0], save_root_dir, 'LICENSE')
    download_file(license_files[1], save_root_dir, 'README')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory to download file")
    parser.add_argument("--resource", type=str,
                        help="Resource name. See RESOURCES_MAP for all possible values")
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print('Please specify resource value. Possible options are:')
        for k, v in RESOURCES_MAP.items():
            print('Resource key={}  description: {}'.format(k, v['desc']))


if __name__ == '__main__':
    main()
