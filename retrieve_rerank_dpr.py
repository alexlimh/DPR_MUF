import argparse
import csv
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from tqdm import tqdm

import numpy as np
import torch

from dpr.data.qa_validation import calculate_matches
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseFlatIndexer,
    DenseReconIndexer
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        args, 
        batch_size: int,
        index: DenseIndexer,
        dist: str = 'original',
        acc_top_k: int = 20,
        rank_top_k: int = 100,
        num_ensemble: int = 20,
        p: float = None,
    ):
        self.args = args
        self.batch_size = batch_size
        self.index = index
        self.dist = dist
        self.acc_top_k = acc_top_k
        self.rank_top_k = rank_top_k
        self.num_ensemble = num_ensemble
        self.p = p

    def get_top_docs(
        self, 
        query_vectors: np.array,
        num_ensemble: int,
        top_docs: int = 100,
    ) -> List[Tuple[List[object], List[float]]]:
        time0 = time.time()
        if len(query_vectors.shape) > 2:
            query_vectors = np.mean(query_vectors[:, :num_ensemble], axis=1)
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        return results
    
    def get_dpr_score_and_conf(self, questions_tensor, all_ctx_ids, question_hits):
        assert (len(questions_tensor.shape) == 3), "Ensemble not valid!"
        results = []
        logits = []
        for question_vector, ctx_ids in tqdm(zip(questions_tensor, all_ctx_ids)):
            ids = []
            mean_scores = []
            all_scores = []
            for ctx_id in ctx_ids:
                ctx_vector = self.index.reconstruct(int(ctx_id))
                score = np.dot(question_vector, ctx_vector)
                ids.append(ctx_id)
                mean_scores.append(np.mean(score[:self.num_ensemble]))
                all_scores.append(list(score))
            logits.append(all_scores[:self.rank_top_k])
            ids, mean_scores = zip(*sorted(list(zip(ids, mean_scores)), key=lambda x: -x[1]))
            results.append((ids, mean_scores))
        
        logits = np.transpose(np.array(logits), [2,0,1]) # K x Q x L

        if self.p is not None:
            confs, probs = self.get_confidence(logits, self.p)
        else:
            best_ece = 9999
            top_k_hits = self.get_top_k_hits(question_hits)
            logger.info("Calibrating using ECE...")
            for p in tqdm(np.arange(0.001,0.011,0.001)):
                confs, probs = self.get_confidence(logits, p)
                ece = self.get_ece(confs, top_k_hits)
                if ece < best_ece:
                    best_p = p
                    best_confs = confs
                    best_ece = ece
            confs = best_confs
            logger.info(f"Best probability for geometric dist: {best_p}, ECE: {best_ece}")
        logger.info(f"Min confidence: {np.min(confs)}, Max confidence: {np.max(confs)}, \
                Mean confidence: {np.mean(confs)}, Median confidence: {np.median(confs)}")

        return results, confs
    
    def get_top_k_hits(self, question_hits):
        top_k_hits = []
        for hits in question_hits:
            top_k_hits.append(float(sum([int(hits[i]) for i in range(self.acc_top_k)]) > 0))
        return top_k_hits

    def get_ece(self, confs, top_k_hits):
        bins = 10
        acc = [[] for i in range(bins)]
        cal_confs = [[] for i in range(bins)]
        conf_ticks = [p for p in np.arange(0,(bins+1)/10,0.1)]
        for conf, top_k_hit in zip(confs, top_k_hits):
            for i in range(bins):
                if conf_ticks[i] <= conf <= conf_ticks[i+1]:
                    acc[i].append(top_k_hit)
                    cal_confs[i].append(conf)
                    break
        acc = [a if len(a) > 0 else [0] for a in acc]
        cal_confs = [p if len(p) > 0 else [0] for p in cal_confs]
        ece = 1./(len(confs)) * sum([len(a) * np.abs(np.mean(a)-np.mean(p)) for a, p in zip(acc, cal_confs)])
        return ece 
        
    def get_confidence(self, logits, p):
        if self.dist == 'exp':
            exp_dist = np.array([p * (1 - p) ** i for i in range(self.rank_top_k)])
            exp_dist = exp_dist / np.sum(exp_dist)

            ranks = np.argsort(-logits, axis=-1)
            probs = []
            for ranks_per_model in ranks:
                probs_per_model = []
                for ranks_per_query in ranks_per_model:
                    probs_per_query = []
                    for rank in ranks_per_query:
                        probs_per_query.append(exp_dist[rank])
                    probs_per_model.append(probs_per_query)
                probs.append(probs_per_model)
            probs = np.array(probs)
        else: # original
            max_logit = np.max(logits, axis=-1, keepdims=True)
            probs = np.exp((logits - max_logit) / p + 1e-6)/(np.sum(np.exp((logits - max_logit) / p + 1e-6), -1, keepdims=True))    
        marg_probs = np.mean(probs, axis=0, keepdims=True)
        marg_ent = -np.sum(marg_probs * np.log(marg_probs+1e-6), axis=-1)
        cond_ent = -np.sum(probs * np.log(probs+1e-6), axis=-1)
        confs = 1. - np.mean(marg_ent - cond_ent, axis=0) / np.log(len(logits))
        return confs, probs


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

def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    confs: List[float] = None,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, (q, conf) in enumerate(zip(questions, confs)):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)
        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "confidence": str(conf),
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )
    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)

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

    logger.info("Loading question embedding ...")
    with open(args.question_embedding, 'rb') as f:
        question_data = pickle.load(f)
    questions_tensor = torch.Tensor(question_data['embedding'])
    

    if args.stage == "first":
        logger.info("===First stage retrieval===")
        all_passages = load_passages(args.ctx_file)
        all_ctx_ids = None

    elif args.stage == "second":
        logger.info("===Second stage retrieval (Rerankig)===")
        ret = load_first_stage_results(args.retrieval_results)
        top_ids_and_scores, all_passages, questions_doc_hits, all_ctx_ids = ret

        questions_doc_hits = validate(
            all_passages,
            question_answers,
            top_ids_and_scores,
            args.validation_workers,
            args.match,
        )

    if len(all_passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )
    
    logger.info("Building index ...")
    vector_size = questions_tensor.size(-1)
    if args.stage == 'first':
        index = DenseFlatIndexer(vector_size, args.index_buffer)
    elif args.stage == 'second':
        index = DenseReconIndexer(vector_size, args.index_buffer)
    retriever = DenseRetriever(args, args.batch_size, index, 
                                dist=args.dist,
                                acc_top_k=args.acc_top_k,
                                rank_top_k=args.rank_top_k,
                                num_ensemble=args.num_ensemble,
                                p=args.p)
    # load_index
    retriever.index.deserialize_from(args.index_path)
    # get top k results
    if args.stage == 'first':
        top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.num_ensemble, args.n_docs)
        confs = [None for _ in top_ids_and_scores]
        questions_doc_hits = validate(
            all_passages,
            question_answers,
            top_ids_and_scores,
            args.validation_workers,
            args.match,
        )
    
    elif args.stage == 'second':  
        _, confs = retriever.get_dpr_score_and_conf(
                                questions_tensor, all_ctx_ids, questions_doc_hits)

    if args.out_file:
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            args.out_file,
            confs
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
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="path to index file",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
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
        help="Batch size for question encoder forward pass",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument(
        "--hnsw_index",
        action="store_true",
        help="If enabled, use inference time efficient HNSW index",
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index"
    )
    parser.add_argument(
        "--question_embedding",
        required=True,
        type=str,
        default=None,
        help="question embedding file",
    )

    parser.add_argument(
        "--retrieval_results",
        type=str,
        default=None,
        help="Retreival results from another model",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="first",
        help="Retreival stage",
    )

    parser.add_argument(
        "--rank_top_k",
        type=int,
        default=100,
        help="Retreival stage",
    )

    parser.add_argument(
        "--acc_top_k",
        type=int,
        default=20,
        help="Retreival stage",
    )
    
    parser.add_argument(
        "--dist",
        type=str,
        default="original",
        help="Retreival distribution",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="temperature coefficient",
    )

    args = parser.parse_args()

    setup_args_gpu(args)
    print_args(args)
    main(args)
