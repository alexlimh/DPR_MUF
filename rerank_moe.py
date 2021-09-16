import argparse
import csv
import json
import logging
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Iterator

import numpy as np
import json
import pathlib

from functools import partial
from multiprocessing import Pool as ProcessPool
from dpr.data.qa_validation import calculate_matches

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config():
    # General.
    config = AttrDict()
    config.models = 'trivia'
    config.dataset = 'nq'
    config.qa_file = 'nq-test.csv'
    config.moe_out_file = 'nq-test-100.json'
    config.in_files = 'nq-test-100.json'
    config.n_docs = 100
    config.validation_workers = 8
    config.match = 'string'
    config.fusion = 'sparse'
    config.seed = 12345
    config.bm25 = False
    config.alpha = None
    return config

def args_type(default):
  if default is None:
    return lambda x: x
  if isinstance(default, bool):
    return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int):
    return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path):
    return lambda x: pathlib.Path(x).expanduser()
  if isinstance(default, (list, tuple)):
    return lambda x: tuple(args_type(default[0])(y) for y in x.split(','))
  return type(default)

def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers

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
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", [top_k_hits[0], 
                                                                         top_k_hits[4],
                                                                         top_k_hits[19],
                                                                         top_k_hits[99]])
    return match_stats.questions_doc_hits

def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    confs: List[int],
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
                "confidence": conf,
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


def get_dpr_score_and_conf(in_file, bm25, alpha):
    with open(in_file) as f:
        data = json.load(f)
    all_passages = {}
    top_ids_and_scores = []
    confs = []
    for sample in data:
        ids = []
        scores = []
        for ctx in sample['ctxs']:
            all_passages[ctx['id']] = (ctx['text'], ctx['title'])
            ids.append(ctx['id'])
            scores.append(float(ctx['score']))
        top_ids_and_scores.append((ids, scores))
        if "bm25" in in_file:
            confs.append(float(alpha) if alpha is not None else alpha)
        else:
            if bm25 or sample['confidence'] == 'None':
                confs.append(1.0)
            else:
                confs.append(float(sample['confidence']))
    return top_ids_and_scores, confs, all_passages


def load_dpr_results(in_files, bm25, alpha):
    logger.info("Loading dpr results ...")
    all_passages = {}
    all_confs = []
    all_top_ids_and_scores = []
    for in_file in tqdm(in_files.split(" ")):
        top_ids_and_scores, confs, passages = get_dpr_score_and_conf(in_file, bm25, alpha)
        all_passages.update(passages)
        all_confs.append(confs)
        all_top_ids_and_scores.append(top_ids_and_scores)
    return all_top_ids_and_scores, all_confs, all_passages

def load_qa_file(qa_file):
    questions = []
    question_answers = []
    for ds_item in parse_qa_csv_file(qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)
    return questions, question_answers

def get_moe_top_ids_and_scores(top_ids_and_scores, confs, default_confs, n_docs):
    ensemble_min_scores = np.min([scores for ctx_ids, scores in top_ids_and_scores], axis=-1)
    id_score_map = {}
    for j, (ctx_ids, scores) in enumerate(top_ids_and_scores):
        for ctx_id, score in zip(ctx_ids, scores):
            if ctx_id not in id_score_map:
                id_score_map[ctx_id] = {}
                id_score_map[ctx_id]['id'] = [j]
                id_score_map[ctx_id]['score'] = [score]
                id_score_map[ctx_id]['conf'] = [confs[j]]
            else:
                id_score_map[ctx_id]['id'].append(j)
                id_score_map[ctx_id]['score'].append(score)
                id_score_map[ctx_id]['conf'].append(confs[j])
    
    moe_scores = []
    moe_ids = []
    for ctx_id, ensemble in id_score_map.items():
        if len(ensemble['conf']) < len(confs):
            for ensemble_id in range(len(confs)):
                if ensemble_id not in ensemble['id']:
                    ensemble['id'].append(ensemble_id)
                    ensemble['score'].append(ensemble_min_scores[ensemble_id])
                    ensemble['conf'].append(confs[ensemble_id])
        weights = np.array(ensemble['conf'])
        weights = weights/np.sum(weights)
        scores = np.array(ensemble['score'])
        moe_scores.append(np.sum(weights * scores))
        moe_ids.append(ctx_id)
    
    moe_ids_and_scores = sorted(list(zip(moe_ids, moe_scores)), key=lambda x: -x[1])[:n_docs]
    moe_top_id_and_score = tuple(zip(*moe_ids_and_scores))
    moe_conf = np.mean(confs)
    return moe_top_id_and_score, moe_conf

def get_moe_score_and_conf(entry, med_confs, fusion, n_docs):
    question, answers, entry = entry
    confs, top_ids_and_scores = zip(*entry)
    if fusion == 'sparse':
        max_index, moe_conf = np.argmax(confs), np.max(confs)
        top_id_and_score = top_ids_and_scores[max_index]
    elif fusion == 'dense': # if not in union, give lowest score
        moe_top_id_and_score, moe_conf = get_moe_top_ids_and_scores(top_ids_and_scores, confs, med_confs, n_docs)
    return [question, answers, moe_top_id_and_score, moe_conf]

def get_moe_results_and_confs(questions, question_answers, all_passages, match, 
                            all_top_ids_and_scores, all_confs, fusion, n_docs, workers_num):
    logger.info("Evalutating Mixture of Experts...")
    med_confs = np.median(np.array(all_confs), axis=-1) # default confidence for missing entry
    # Parallel reranking
    processes = ProcessPool(processes=workers_num)
    entries = [(questions[i], question_answers[i], [(confs[i], (top_ids_and_scores[i][0], top_ids_and_scores[i][1])) \
                                for confs, top_ids_and_scores in zip(all_confs, all_top_ids_and_scores)]) \
                                    for i in range(len(all_confs[0]))]
    get_score_partial = partial(
        get_moe_score_and_conf, med_confs=med_confs, fusion=fusion, n_docs=n_docs
    )
    results = list(tqdm(processes.imap(get_score_partial, entries), total=len(all_confs[0])))
    processes.close()
    questions, question_answers, moe_top_ids_and_scores, moe_confs = zip(*results)
    return questions, question_answers, moe_top_ids_and_scores, moe_confs


def validate_and_save_results(all_passages, questions, question_answers, validation_workers,
                                match, out_file, top_ids_and_scores, confs):
    questions_doc_hits = validate(
        all_passages,
        question_answers,
        top_ids_and_scores,
        validation_workers,
        match,
    )
    if out_file:
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            out_file,
            confs,
        )


def main(args):
    print(args)
    random.seed(args.seed)
    all_top_ids_and_scores, all_confs, all_passages = load_dpr_results(args.in_files, args.bm25, args.alpha)
    questions, question_answers = load_qa_file(args.qa_file)
    questions, question_answers, moe_top_ids_and_scores, moe_confs = get_moe_results_and_confs(
                        questions, question_answers, all_passages, args.match,
                        all_top_ids_and_scores, all_confs, args.fusion, 
                        args.n_docs, args.validation_workers)
    logger.info("Validating mixture of experts...")
    validate_and_save_results(all_passages, questions, question_answers, args.validation_workers,
                                args.match, args.moe_out_file, moe_top_ids_and_scores, moe_confs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)
    main(parser.parse_args())
