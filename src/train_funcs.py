#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:37:26 2019.

@author: weetee
"""
import logging
from itertools import combinations

import torch
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)


class Two_Headed_Loss(nn.Module):
    """
    Implements LM Loss and matching-the-blanks loss concurrently.
    """

    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(
            ignore_index=self.lm_ignore_idx, reduction="sum"
        )
        self.use_logits = use_logits
        self.normalize = normalize

        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction="sum")
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1 / (torch.norm(f1_vec) * torch.norm(f2_vec))
        else:
            factor = 1.0

        if not self.use_logits:
            p = 1 / (1 + torch.exp(-factor * torch.dot(f1_vec, f2_vec)))
        else:
            p = factor * torch.dot(f1_vec, f2_vec)
        return p

    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels):
        """
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, embeddings)
        blank_labels: (batch_size, 0 or 1)
        """
        pos_idxs = [
            i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1
        ]
        neg_idxs = [
            i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0
        ]

        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(
                    self.p_(blank_logits[pos1, :], blank_logits[pos2, :])
                )
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
            if blank_logits.is_cuda:
                pos_logits = pos_logits.cuda()

        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(
                    self.p_(blank_logits[pos_idx, :], blank_logits[neg_idx, :])
                )
        neg_logits = torch.stack(neg_logits, dim=0)
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]

        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)

        if blank_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()

        lm_loss = self.LM_criterion(lm_logits, lm_labels)

        blank_loss = self.BCE_criterion(
            torch.cat([pos_logits, neg_logits], dim=0), blank_labels_
        )

        total_loss = lm_loss + blank_loss
        return total_loss
