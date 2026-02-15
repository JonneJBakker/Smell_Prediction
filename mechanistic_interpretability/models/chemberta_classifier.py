import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, classification_report,
    accuracy_score, f1_score, hamming_loss,
    jaccard_score, roc_auc_score
)
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    EarlyStoppingCallback,
    RobertaModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.simple_mlp import SimpleMLP

DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"

class FocalLoss(nn.Module):
    """
    Multi-label focal loss.
    Args:
        alpha: balancing factor (float or tensor of shape [num_labels])
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # p_t = sigmoid(logit) if target=1, else 1-sigmoid(logit)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # focal weight
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden_states: (batch, seq_len, hidden)
    # attention_mask: (batch, seq_len)
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)                  # (batch, hidden)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

class ChembertaMultiLabelClassifier(nn.Module):
    """
    ChemBERTa multi-label classification model.
    """

    def __init__(
        self,
        pretrained,
        num_labels,
        num_features=0,
        dropout=0.3,
        hidden_channels=100,
        num_mlp_layers=1,
        pos_weight=None,
        gamma = 0.75,
        alpha = None,
        pooling_strat = "mean",
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)
        self.pooling_strat = pooling_strat


        #freeze language model
        for param in self.roberta.parameters():
             param.requires_grad = False


        # If we want attention pooling
        if self.pooling_strat == "attention":
            self.query_vector = nn.Parameter(
                torch.randn(self.roberta.config.hidden_size)
            )


        self.dropout = nn.Dropout(dropout)
        if self.pooling_strat == "cls_mean" or self.pooling_strat == "max_mean":
            num_input_features = 2*self.roberta.config.hidden_size
        else:
            num_input_features = self.roberta.config.hidden_size

        # Output dimension = num_labels, one logit per label
        self.mlp = SimpleMLP(
            num_input_features,
            hidden_channels,
            num_mlp_layers,
            num_labels,
            dropout,
        )


        self.loss_fct = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
        )

        ''''
        if pos_weight is not None:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()
        '''''
    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None, strat=None):
        if strat is None:
            strat = self.pooling_strat
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        if strat == "cls_mean":
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
            token_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)
            mean_pooled = mean_pool(token_embs, attention_mask)  # (batch, hidden)
            pooled = torch.cat([mean_pooled, cls_emb], dim=1)
            x = self.dropout(pooled)

        if strat == "max_mean":
            token_embs = outputs.last_hidden_state
            mean_pooled = mean_pool(token_embs, attention_mask)  # [B, H]

            mask = attention_mask.unsqueeze(-1).bool()  # [B, N, 1]

            # set padded positions to -inf so they don't affect max
            masked_token_embs = token_embs.masked_fill(~mask, float("-inf"))

            max_pooled, _ = masked_token_embs.max(dim=1)  # [B, H]

            pooled = torch.cat([mean_pooled, max_pooled], dim=1)  # [B, 2H]
            x = self.dropout(pooled)

        if strat == "mean":
            token_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)
            pooled = mean_pool(token_embs, attention_mask)  # (batch, hidden)
            x = self.dropout(pooled)

        elif strat == "cls":
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
            x = self.dropout(cls_emb)

        elif strat == "attention":
            token_embs = outputs.last_hidden_state  # [B, N, D]

            # [B, N]
            attn_scores = torch.matmul(token_embs, self.query_vector)

            # [B, N, 1]
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

            # weighted sum → [B, D]
            pooled = torch.sum(token_embs * attn_weights, dim=1)

            x = self.dropout(pooled)

        # (batch_size, num_labels)
        logits = self.mlp(x)

        loss = None
        if labels is not None:
            # labels expected shape: (batch_size, num_labels), dtype=float
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )