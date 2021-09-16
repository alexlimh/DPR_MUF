import argparse
import csv
import json
import gzip
import logging
from typing import List, Tuple, Dict, Iterator
from tqdm import tqdm

from dpr.data.qa_validation import calculate_matches
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    add_tokenizer_params,
    add_cuda_params,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs

def load_first_stage_results(retrieval_results):
    with open(retrieval_results) as f:
        retrieved_data = json.load(f)
            
    all_passages = {}
    all_ctx_ids = []
    questions_doc_hits = []
    top_ids_and_scores = []
    for sample in retrieved_data:
        ctxs = sample['ctxs']
        ids = []
        hit = []
        scores = []
        for ctx in ctxs:
            ids.append(ctx['id'])
            scores.append(ctx['score'])
            all_passages[ctx['id']] = (ctx['text'], ctx['title'])
            hit.append(ctx["has_answer"])
        all_ctx_ids.append(ids)
        questions_doc_hits.append(hit)
        top_ids_and_scores.append((ids, scores))
    
    return top_ids_and_scores, all_passages, questions_doc_hits, all_ctx_ids

def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def main(args):
    questions = []
    question_answers = []
    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    logger.info("===Second stage retrieval (Rerankig)===")
    ret = load_first_stage_results(args.retrieval_results)
    top_ids_and_scores, all_passages, questions_doc_hits, all_ctx_ids = ret

    # index all passages
    validate(
        all_passages,
        question_answers,
        top_ids_and_scores,
        args.validation_workers,
        args.match,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--qa_file",
        required=True,
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    
    
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    
    parser.add_argument(
        "--n-docs", type=int, default=200, help="Amount of top docs to return"
    )
    
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for question encoder forward pass",
    )
    parser.add_argument(
        "--num_ensemble",
        type=int,
        default=100,
        help="number of ensemble members",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )

    parser.add_argument(
        "--retrieval_results",
        type=str,
        default=None,
        help="Retreival results from the first-stage model",
    )

    args = parser.parse_args()

    setup_args_gpu(args)
    print_args(args)
    main(args)
