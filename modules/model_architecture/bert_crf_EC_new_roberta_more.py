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
    
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gate_values = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Weighted combination of expert outputs
        return torch.sum(gate_values.unsqueeze(-1) * expert_outputs, dim=1)


class Roberta_token_classification(RobertaPreTrainedModel):
    def __init__(self, config, num_labels_=2, auxnum_labels=2, num_experts = 8):
        super(Roberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        self.dropout = WordDropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.moe = MoE(config.hidden_size, config.hidden_size, num_experts)
        self.crf = CRF(num_labels_, batch_first=True)
        self.init_weights()

    def forward(self, input_ids_text, segment_ids_text, input_mask_text, input_ids_img, segment_ids_img, input_mask_img, input_ids_origin, segment_ids_origin, input_mask_origin, image_features, labels_text=None, labels_img=None, labels_origin = None):
        features = self.roberta(input_ids_img, attention_mask=input_mask_img)
        sequence_output_img = features["last_hidden_state"]
        sequence_output_img = self.dropout(sequence_output_img)

        # features_2 = self.roberta(input_ids_text, attention_mask=input_mask_text)
        # sequence_output_text = features_2["last_hidden_state"]
        # sequence_output_text = self.dropout(sequence_output_text)

        fusion_features = torch.cat([sequence_output_img, image_features], dim=-1)
        p_T = torch.sigmoid(self.gating_layer(fusion_features))  # Pθc(T|x, I)
        p_I = 1 - p_T  # Pθc(I|x, I)

        # Fuse Probabilities
        combined_probs = p_T * sequence_output_img + p_I * image_features  # P(y|x, I)

        if labels_text is not None:
            features_origin = self.roberta(input_ids_origin,attention_mask=input_mask_origin, token_type_ids=segment_ids_origin)
            sequence_output_origin = features_origin["last_hidden_state"]
            sequence_output_origin = self.dropout(sequence_output_origin)
            logits_origin = self.classifier(sequence_output_origin)

            logits_img = self.classifier(sequence_output_img)
            # logits_text = self.classifier(sequence_output_text)
            loss_img = -self.crf(logits_img, labels_img, mask=input_mask_img.byte(), reduction='mean')
            # loss_text = -self.crf(logits_text, labels_text, mask=input_mask_text.byte(), reduction='mean')
            loss_origin = -self.crf(logits_origin, labels_origin, mask=input_mask_origin.byte(), reduction='mean')
            # main_loss = 0.33 * (loss_img + loss_text + loss_origin)
            main_loss = 0.5 * (loss_img + loss_origin)
            return main_loss
        else:
            pred_tags = self.crf.decode(logits_img, mask=input_mask_img.byte())
            return pred_tags


if __name__ == "__main__":
    model = Roberta_token_classification.from_pretrained(r'vinai/phobert-base-v2', cache_dir="cache")
    print(model)