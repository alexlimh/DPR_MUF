import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
import pandas as pd

from dpr.models import init_batchensemble_components
import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    add_training_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def generate_question_ensemble_vectors(batch_size, 
                                model,
                                question_vectors) -> T:
    n = len(question_vectors)
    bsz = batch_size
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):

            batch_question_vectors = torch.stack([torch.from_numpy(q)
                for q in question_vectors[batch_start : batch_start + bsz]
            ], dim=0).cuda()
            out = model(batch_question_vectors)
            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)

    logger.info("Total encoded queries tensor %s", query_tensor.size())

    assert query_tensor.size(0) == len(question_vectors)
    return query_tensor


def generate_question_vectors(batch_size, 
                                questions, 
                                question_encoder,
                                tensorizer) -> T:
    n = len(questions)
    bsz = batch_size
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):

            batch_token_tensors = [
                tensorizer.text_to_tensor(q)
                for q in questions[batch_start : batch_start + bsz]
            ]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)
            _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)

    logger.info("Total encoded queries tensor %s", query_tensor.size())

    assert query_tensor.size(0) == len(questions)
    return query_tensor

def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers

def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector

def main(args):
    if args.batch_ensemble:
        with open(args.question_embedding, "rb") as f:
            data = pickle.load(f)
        question_embedding = data['embedding']
        args.vector_size = question_embedding[0].shape[-1]

        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)
        
        model, _ = init_batchensemble_components(
            args.encoder_model_type, args
        )
        model.eval()
        
        # load weights from the model file
        model_to_load = get_model_obj(model)
        logger.info("Loading saved model state ...")

        prefix_len = len("question_model.")
        saved_state = {
            key: value
            for (key, value) in saved_state.model_dict.items()
        }
        model_to_load.load_state_dict(saved_state)
        
        model.to("cuda")
        questions_tensor = generate_question_ensemble_vectors(args.batch_size, model, question_embedding)
    
    else:
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, encoder, _ = init_biencoder_components(
            args.encoder_model_type, args, inference_only=True
        )

        question_encoder = encoder.question_model

        question_encoder, _ = setup_for_distributed_mode(
            question_encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
        )
        question_encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(question_encoder)
        logger.info("Loading saved model state ...")

        prefix_len = len("question_model.")
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith("question_model.")
        }
        model_to_load.load_state_dict(question_encoder_state)
        
        # get questions & answers
        questions = []
        question_answers = []
        
        file_type = args.qa_file.split(".")[-1]
        if file_type == "csv":
            for ds_item in parse_qa_csv_file(args.qa_file):
                question, answers = ds_item
                questions.append(question)
                question_answers.append(answers)
        else:
            with open(args.qa_file) as f:
                data = json.load(f)
            for sample in data:
                if len(sample["positive_ctxs"]) > 0:
                    questions.append(sample["question"])
                    question_answers.append(sample["answers"])
        # generate vectors
        questions_tensor = generate_question_vectors(args.batch_size, questions, question_encoder, tensorizer)
        
    # save vectors
    embeddings = {'id': [], 'embedding': []}
    for i, embedding in enumerate(questions_tensor.numpy()):        
        embeddings['id'].append(i)
        embeddings['embedding'].append(embedding)
    
    with open(args.out_file, "wb") as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)

    parser.add_argument(
        "--qa_file",
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument('--question_embedding', type=str, help='path to first-stage query embeddings')
    parser.add_argument('--out_file', type=str, help='path to store query embeddings', required=True)
    args = parser.parse_args()

    setup_args_gpu(args)
    print_args(args)
    main(args)
    