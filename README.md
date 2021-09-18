# Multi-Task Dense Retrieval via Model Uncertainty Fusionfor Open-Domain Question Answering

This repo provides a model uncertainty fusion (MUF) solution to multi-task question answering based on [DPR](https://github.com/facebookresearch/DPR). Instead of training a single DPR model on the union of all datasets, we simply train different DPR experts for different tasks and aggregate their predictions using ensemble uncertainty estimation, which allows the users to continually add new models to the current pool. For more details please refer to [our paper]() (upcoming in EMNLP2021).

Comparison between DPR-multi (joint training) and DPR-MUF on five QA tasks:

| Top-100 accuracy  | NQ  | Trivia  | WQ | Curated  | SQuAD |
|---|---|---|---|---|---|
|DPR |85.9 |84.5 |80.2 |92.2 |76.8
DPR-Multi (w/o SQuAD) |86.1 |**84.8** |83.0 |93.4  |67.7 
DPR-MUF (w/o SQuAD) | **86.5** |84.4 | 83.9 |94.7 | 72.0 
DPR-MUF | 86.4 |  84.7 | **84.0** | **95.0** | **78.3**

According to the [DPR paper](https://arxiv.org/abs/2004.04906), "SQuAD is limited to a small set of Wikipedia documents
and thus introduces unwanted bias", and therefore SQuAD is not included in joint-training. In contrast, our MUF solution that incorporates the expert on SQuAD further improves the accuracy.

## Installation
```
git clone https://github.com/alexlimh/DPR_MUF.git
cd DPR_MUF
pip install .
```
The requirement is the same as the DPR repo's.

## Data, Checkpoints, and Index Files 
```
python data/download_data.py \
	--resource {key from download_data.py's RESOURCES_MAP}  \
	[optional --output_dir {your location}]
```

The available resources include:
- train/dev/test files;
- dpr ranking results with confidence (i.e., 1-uncertainty);
- pre-trained DPR ckpts;
- pre-trained ensemble ckpts;
- pre-built faiss flat index (for search) and IVF index (for indexing);

for NQ, Trivia, Curated, WQ, and SQuAD datasets. 
Notes:
- To avoid violation to SQuAD's policy, we **do not** provide SQuAD's test set and its retrieval results. However, we do provide its model checkpoints and indexes for users to reproduce our results.
- As for other resources and data (e.g., reader ckpts) please refer to the DPR repo.

For example:
```
python data/download_data.py --resource data.wikipedia_split.psgs_w100 # corpus
python data/download_data.py --resource data.retriever.nq # train and dev file
python data/download_data.py --resource data.retriever.qas.nq # train/dev/test file
python data/download_data.py --resource indexes.single.nq # index file
python data/download_data.py --resource data.retriever_results.nq.single.curated.test # retrieval results on CuratedTrec using DPR-NQ model 
```


## 1. Retriever Training
In the following, we take NQ dataset as an example:
```
python -m torch.distributed.launch \
	--nproc_per_node=8 train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--seed 12345 \
	--sequence_length 256 \
	--warmup_steps 1237 \
	--batch_size 16 \
	--do_lower_case \
	--train_file "{glob expression to train files downloaded as 'data.retriever.nq-train' resource}" \
	--dev_file "{glob expression to dev files downloaded as 'data.retriever.nq-dev' resource}" \
	--output_dir {your output dir} \
	--learning_rate 2e-05 \
	--num_train_epochs 40 \
	--dev_batch_size 16 \
	--val_av_rank_start_epoch 30
```
For SQuAD we found that the best learning rate is `1e-6`. The other hyperparameters are the same for the others datasets. If you prefer not to train the DPR model from scratch, you can use the checkpoints we provide in the `data/download_data.py`. For more training details please refer to the original DPR repo.

## 2. Encoding Queries and Passages
Before learning the uncertainty, we pre-encoded the DPR's representations for both queries and passages. We save the question embeddings in `.pkl` file. 

Encode queries for train/dev/test:
```
python encode_queries.py \
 			--qa_file {path to train/dev/test qas file} \
 			--model_file {path to checkpoint file from step 1} \
 			--encoder_model_type hf_bert \
			--out_file {your output dir from step 1}/{train/dev/test query embedding pkl file}
```

Encode passage embeddings:
```
python generate_dense_embeddings.py \
   --model_file {path to checkpoint from step 1} \
   --ctx_file {path to psgs_w100.tsv file} \
   --shard_id {shard_num, 0-based} --num_shards {total number of shards} \
   --out_file ${out files location + name PREFX}
```
This step can also be skipped if you used our pre-encoded queries which are provided in the data download script.

## 3. Building index
Build `flat` index:
```
python build_index.py --encoded_ctx_file "{path to generated passage embeddings in step 2}" \
	--index_path {path to your flat index dir} \
	--index_type flat
```

Build `ivf` index for fast indexing passages using passage id:
```
python build_index.py --encoded_ctx_file "{path to generated passage embeddings in step 2}" \
	--index_path {path to your recon index dir} \
	--index_type ivf
```
The pre-built index are provided in the data download script.

## 4. Retrieval (first-stage)
We first need to retrieve the passages using DPR and compute the uncertainty score based on that.
```
python retrieve_rerank_dpr.py \
		--ctx_file {path to psgs_w100.tsv file} \
		--qa_file {path to test qas file} \
		--index_path {path to your **flat** index file from step 3} \
		--out_file  {path for saving your ranking results} \
		--question_embedding {test question embeddings from step 2} \
		--n-docs 1000 \
		--validation_workers 16 \
		--save_or_load_index \
		--match $match # "regex" for curated and "string" for the other \
		--stage first
```
The first-stage ranking results (with uncertainty) is provided as well.


## 5. Ensemble Training
To compute the uncertainty of each expert, we fix the DPR's representations and train an ensemble of small neural networks, where the mutual information between the ensemble parameters and the predictions will be used as uncertainty. Ensemble training is fast and only requires a single GPU.
```
python train_batch_ensemble.py \
			--output_dir {your_output_dir} \
			--max_grad_norm 2.0 \
			--encoder_model_type hf_bert \
			--seed 12345 \
			--batch_size 128 \
			--do_lower_case \
			--train_file "{glob expression to train files downloaded as 'data.retriever.nq-train' resource}" \
            --dev_file "{glob expression to dev files downloaded as 'data.retriever.nq-dev' resource}" \
            --learning_rate 2e-5 \
			--num_train_epochs 100 \
			--val_av_rank_start_epoch 1000 \
			--dev_batch_size 32 \
			--hard_negatives 1
            --ensemble_size 20 \
			--train_question_embedding {path to your encoded train embeddings from step 2} \
			--dev_question_embedding {path to your encoded dev embeddings from step 2} \
			--encoded_ctx_file {path to your **ivf** index from step 3} \
			--save_or_load_index
```
Notes:
- We set `num_train_epochs=100` and `val_av_rank_start_epoch=1000` to avoid rank validation as we do not care too much about the ensemble's ranking accuracy but actually want to overfit to the training domain to get accurate uncertainty prediction. For more details please refer to our paper.
- The trained ensemble ckpt is provided in the repo as well.

## 6. Uncertainty Estimation
We need to encode the ensemble embeddings for queries:
```
python encode_queries.py \
			--model_file {your output dir from step 5}/{the best ckpt} \
			--encoder_model_type hf_bert \
			--batch_ensemble \
			--question_embedding {path to your dpr test query embedding from step 2} \
			--ensemble_size 20 \
			--out_file {path for saving your ensemble query embedding}
```

Next, we compute the uncertainty for the ranking results from step 4.
```
python retrieve_rerank_dpr.py \
                --ctx_file {path to psgs_w100.tsv file} \
                --qa_file {path to test qas file} \
                --index_path {path to your **recon** index file from step 3} \
		        --out_file {path for saving your uncertainty + ranking results} \
                --retrieval_results {path to your **first-stage** ranking results from step 4}  \
                --question_embedding {ensemble test question embeddings from the previous command} \
                --validation_workers 16 \
                --num_ensemble 20 \
                --n-docs 1000 # first stage \
                --rank_top_k 1000 # for uncertainty \
                --acc_top_k 1 # calibration target \
                --dist original \
                --p 1e-3 # if set None then it would search the best p automatically \
                --stage second
```
Notes:
- `rank_top_k`: top-k documents for calculating the uncertainty
- `acc_top_k`: for uncertainty calibration (accuracy for ECE)
- `dist`: use the softmax or geometric distribution for calulating the probablity p(d|q). `original` is the softmax distribution.
- `p`: temperature coefficient for tuning the sharpness of the softmax distribution. We suggest using `p=1e-3` for best performance.


## 7. Model Fusion
After repeating the above step for each expert, we could calculate the model fusion results on your target datasets. We use NQ as an example:
```
in_files="{nq expert's output file from step 6} \
{trivia expert's output file from step 6} \
{wq expert's output file from step 6} \
{curated expert's output file from step 6} \
{squad1 expert's output file from step 6}"
     
python rerank_moe.py --models "nq+trivia+wq+curated+squad1" \
        --dataset nq \
        --moe_out_file {path for saving your ranking results} \
        --in_files "${in_files}" \
        --qa_file {path to test qas file} \
        --match $match # "string" or "regex" \
        --fusion dense \
        --validation_workers 16 \
        --n_docs 100
```
Notes:
- `fusion`: `dense` or `sparse`. `dense` means taking the weighted sum of all experts' scores. `sparse` means taking the max score of the experts.

## Other scripts
If you want to evaluate your retrieval results directly, you can use the following scripts:
```
python eval_retrieval_results.py \
    --qa_file {path to test qas file} \
    --retrieval_results {path to retrieval results file} \
    --n-docs 100 \
    --validation_workers 16 \
    --match $match # "regex" for curated and "string" for the other
```

## Misc.
Some variable names and classes might be confusing to the readers:
- `batch ensemble`: This refers to the [BatchEnsemble paper](https://arxiv.org/abs/2002.06715) where they use rank-1 matrices for the ensemble built on a single neural network. Inspired by their work, we train an ensemble of small neural networks for uncertainty prediction, which we refer to as `batch_ensemble` in the code.

- `recon_index`: This inherits the `IndexIVFFlat` class from faiss where users could efficiently access the vector representation using its id. The name `recon` simply stands for reconstruction as the indexing function is `index.reconstruct(ctx_id)`. Although the `flat` index also has the reconstruction method, it's much slower than the `ivf` index.

- `moe`: This stands for _mixture-of-experts_ which is essentially the model fusion method, where we replace the learned gating function with individual uncertainty estimation. 

## Reference

If you find the codes and data useful, please cite this paper (upcoming in EMNLP2021):
```
Coming soon!
```

and also the original DPR paper:
```
@misc{karpukhin2020dense,  
    title={Dense Passage Retrieval for Open-Domain Question Answering},  
    author={Vladimir Karpukhin and Barlas Oğuz and Sewon Min and Patrick Lewis and Ledell Wu and Sergey Edunov and Danqi Chen and Wen-tau Yih},  
    year={2020},  
    eprint={2004.04906},  
    archivePrefix={arXiv},  
    primaryClass={cs.CL}  
}  
```

## License
DPR_MUF inherits DPR's CC-BY-NC 4.0 licensed as of now.