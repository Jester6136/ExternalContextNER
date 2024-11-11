from transformers import RobertaModel, RobertaPreTrainedModel
from torch import nn as nn
from torchcrf import CRF
import torch
import torch.nn.functional as F  # For softmax


class WordDropout(nn.Module):
    def __init__(self, dropout_rate: float = 0.1):
        super(WordDropout, self).__init__()
        assert 0.0 <= dropout_rate < 1.0, '0.0 <= dropout rate < 1.0 must be satisfied!'
        self.dropout_rate = dropout_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.dropout_rate:
            return inputs

        mask = inputs.new_empty(*inputs.shape[:2], 1, requires_grad=False).bernoulli_(
            1.0 - self.dropout_rate
        )
        mask = mask.expand_as(inputs)
        return inputs * mask
    
class Roberta_token_classification(RobertaPreTrainedModel):
    def __init__(self, config, num_labels_=2, auxnum_labels=2):
        super(Roberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        self.dropout = WordDropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2, labels=None, labels2 = None):
        features = self.roberta(input_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
    
        if labels is not None:
            features2 = self.roberta(input_ids2,attention_mask=input_mask2, token_type_ids=segment_ids2)
            sequence_output2 = features2["last_hidden_state"]
            sequence_output2 = self.dropout(sequence_output2)
            logits2 = self.classifier(sequence_output2)

            loss1 = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
            loss2 = -self.crf(logits2, labels2, mask=input_mask2.byte(), reduction='mean')
            main_loss = 0.5 * (loss1 + loss2)
            return main_loss
        else:
            pred_tags = self.crf.decode(logits, mask=input_mask.byte())
            return pred_tags


if __name__ == "__main__":
    model = Roberta_token_classification.from_pretrained(r'D:\ExternalContextNER\cache\pubmedRoberta')
    print(model)