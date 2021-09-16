import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.utils.data_utils import Tensorizer
from dpr.utils.data_utils import normalize_question

logger = logging.getLogger(__name__)

EnsembleBatch = collections.namedtuple(
    "BatchEnsembleInput",
    [
        "question_ids",
        "context_ids",
        "is_positive",
    ],
)

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


class BatchEnsemble(nn.Module):
    def __init__(
        self,
        vector_size: int = 768,
        ensemble_size: int = 10, 
        layers: int = 2,
        units: int = 512, 
        activation: str = 'relu',
        device: str = 'cuda',
        anchor_prior: bool = False,
    ):
        super(BatchEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.dim = vector_size
        # ensemble nn
        self.anchor_prior = anchor_prior
        self.build_ensemble(ensemble_size, vector_size, layers+1, units, activation)
    
    def weights_init(self, m):
        std = (20/self.dim)**0.5
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0., std=std)
            torch.nn.init.normal_(m.bias, mean=0., std=std)

    def build_ensemble(self, ensemble_size, dim, layers, units, activation):
        act_dict = dict(relu=nn.ReLU, sigmoid=nn.Sigmoid, tanh=nn.Tanh)
        self.ensemble = []
        self.ensemble_init_params = []
        for i in range(ensemble_size):
            net = []
            for j in range(layers):
                if j == 0:
                    net.extend([nn.Linear(dim, units), act_dict[activation]()])
                elif j == layers - 1:
                    net.extend([nn.Linear(units, dim)])
                else:
                    net.extend([nn.Linear(units, units), act_dict[activation]()])
            
            net = nn.Sequential(*net)
            if self.anchor_prior:
                net.apply(lambda m: self.weights_init(m))
                self.ensemble_init_params.append([p.detach().cuda() for p in net.parameters()])
            self.ensemble.append(net)
                
        self.ensemble = nn.ModuleList(self.ensemble)
        
    def forward(
        self,
        q_vector,
    ) -> Tuple[T, T]:
        q_vectors = []
        for net in self.ensemble:
            q_vectors.append(net(q_vector))
        q_vector = torch.stack(q_vectors, dim=1)
        return q_vector

    @classmethod
    def create_batchensemble_input(
        cls,
        ensemble_size: int,
        samples: List,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
    ) -> EnsembleBatch:
    
        question_ids = [sample['question_id'] for sample in samples]
        positive_ctx_indices = []
        ctx_ids = []
        batch_size = len(samples) // ensemble_size
        for i, sample in enumerate(samples):
            if i % batch_size == 0:
                ctx_len = len(ctx_ids)

            positive_ctx_indices.append(len(ctx_ids) - ctx_len)

            if shuffle and shuffle_positives:
                positive_ctxs = sample["positive_ctxs"]
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample["positive_ctxs"][0]
            
            neg_ctxs = sample["negative_ctxs"]
            hard_neg_ctxs = sample["hard_negative_ctxs"]
            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)
            
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives] # the hard negatives could be empty
            neg_ctxs = neg_ctxs[0:num_other_negatives + num_hard_negatives]
            all_ctxs = [positive_ctx] + (hard_neg_ctxs + neg_ctxs)[:num_hard_negatives]
            ctx_ids.extend([ctx["passage_id"] if "passage_id" in ctx else ctx["psg_id"] for ctx in all_ctxs])

        return EnsembleBatch(
            question_ids,
            ctx_ids,
            positive_ctx_indices,
        )


class BatchEnsembleNllLoss(object):
    def calc(
        self,
        ensemble_q_vectors: T,
        ensemble_ctx_vectors: T,
        ensemble_positive_idx_per_question: list,
    ) -> Tuple[T, int]:
        ensemble_loss = 0.
        correct_predictions_count = 0
        ensemble_size = ensemble_q_vectors.size(1)
        q_num = ensemble_q_vectors.size(0) // ensemble_size
        ctx_num = ensemble_ctx_vectors.size(0) // ensemble_size
        
        for i in range(ensemble_size):
            positive_idx_per_question = ensemble_positive_idx_per_question[i*q_num:(i+1)*q_num]
            q_vectors = ensemble_q_vectors[i*q_num:(i+1)*q_num, i, :]
            ctx_vectors = ensemble_ctx_vectors[i*ctx_num:(i+1)*ctx_num]
            
            scores = self.get_scores(q_vectors, ctx_vectors)
            softmax_scores = F.log_softmax(scores, dim=1)
            labels = torch.tensor(positive_idx_per_question).to(softmax_scores.device)
            ensemble_loss += F.nll_loss(
                softmax_scores,
                labels,
                reduction="mean",
            )

            max_score, max_idxs = torch.max(softmax_scores, 1)
            correct_predictions_count += (
                max_idxs == labels
            ).sum().double()

        ensemble_loss /= ensemble_size
        correct_predictions_count /= ensemble_size
        return ensemble_loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        return dot_product_scores(q_vector, ctx_vectors)
