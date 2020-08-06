"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for HM model
"""
from torch import nn
from torch.nn import functional as F

from .model import UniterPreTrainedModel, UniterModel


class UniterForHm(UniterPreTrainedModel):
    def __init__(self, config, img_dim):
        super().__init__(config)

        self.uniter = UniterModel(config, img_dim)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.dense(pooled_output)
        logits = self.dropout(logits)
        logits = self.classifier(logits).squeeze(1)

        if compute_loss:
            hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            return hm_loss
        else:
            return logits
