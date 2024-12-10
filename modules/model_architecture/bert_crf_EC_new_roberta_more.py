from transformers import RobertaModel, RobertaPreTrainedModel
from torch import nn as nn
import copy
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
    
'''source https://github.com/SimiaoZuo/MoEBERT/blob/master/src/transformers/moebert/moe_layer.py'''
class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert, route_method, vocab_size, hash_list):
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.route_method = route_method
        if route_method in ["gate-token", "gate-sentence"]:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
        elif route_method == "hash-random":
            self.hash_list = self._random_hash_list(vocab_size)
        elif route_method == "hash-balance":
            self.hash_list = self._balance_hash_list(hash_list)
        else:
            raise KeyError("Routing method not supported.")

    def _random_hash_list(self, vocab_size):
        hash_list = torch.randint(low=0, high=self.num_experts, size=(vocab_size,))
        return hash_list

    def _balance_hash_list(self, hash_list):
        with open(hash_list, "rb") as file:
            result = pickle.load(file)
        result = torch.tensor(result, dtype=torch.int64)
        return result

    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        logits_gate = self.gate(x)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x
            return input_x

        x = [forward_expert(x[i], prob_gate[i], i) for i in range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)

        return x, balance_loss, gate_load

    def _forward_gate_sentence(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_sentences = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_sentences.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_sentences.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_sentences.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_sentences.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x.unsqueeze(-1)
            return input_x

        result = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                result.append(forward_expert(x[i], prob_gate[i], i))
        result = torch.vstack(result)
        result = result[order.argsort(0)]  # restore original order

        return result, balance_loss, gate_load

    def _forward_sentence_single_expert(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        gate_load = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        x = self.experts[gate.cpu().item()].forward(x)
        return x, 0.0, gate_load

    def _forward_hash(self, x, input_ids):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        self.hash_list = self.hash_list.to(x.device)
        gate = self.hash_list[input_ids.view(-1)]

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        x = [self.experts[i].forward(x[i]) for i in range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)

        return x, 0.0, gate_load

    def forward(self, x, input_ids, attention_mask):
        if self.route_method == "gate-token":
            x, balance_loss, gate_load = self._forward_gate_token(x)
        elif self.route_method == "gate-sentence":
            if x.size(0) == 1:
                x, balance_loss, gate_load = self._forward_sentence_single_expert(x, attention_mask)
            else:
                x, balance_loss, gate_load = self._forward_gate_sentence(x, attention_mask)
        elif self.route_method in ["hash-random", "hash-balance"]:
            x, balance_loss, gate_load = self._forward_hash(x, input_ids)
        else:
            raise KeyError("Routing method not supported.")

        return x, balance_loss, gate_load
class Expert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)  # Fully connected layer
        self.activation = nn.ReLU()  # Non-linear activation

    def forward(self, x):
        return self.activation(self.fc(x))


class Roberta_token_classification(RobertaPreTrainedModel):
    def __init__(self, config, num_labels_=14, auxnum_labels=2, num_experts=8):
        super(Roberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)  # Text Encoder
        # self.dropout = WordDropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size + 512, config.hidden_size)
        # Initialize the MoE Layer
        self.moe = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=5,
            expert=Expert(hidden_size=config.hidden_size),
            route_method='gate-token',
            vocab_size=config.vocab_size,
            hash_list=None
        )
        self.crf = CRF(num_labels_, batch_first=True)
        
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.init_weights()

    def forward(
        self, 
        input_ids_text, segment_ids_text, input_mask_text, 
        input_ids_img, segment_ids_img, input_mask_img, 
        input_ids_origin, segment_ids_origin, input_mask_origin, 
        image_features, labels_text=None, labels_img=None, labels_origin=None
    ):
        text_features = self.roberta(input_ids_origin, attention_mask=input_mask_origin)
        rT = text_features["last_hidden_state"][:, 0, :]  # CLS token
        rT = self.dropout(rT)
        rI = image_features.squeeze()  # Image representation rI
        # Calculate P(T|x,I)
        fusion_features = torch.cat([rT, rI], dim=-1)
        p_T = torch.sigmoid(self.linear(fusion_features))  # Pθc(T|x, I)
        p_I = 1 - p_T  # Pθc(I|x, I)
        
        
        features_ec_img = self.roberta(input_ids_img, attention_mask=input_mask_img)
        sequence_output_ec_img = features_ec_img["last_hidden_state"]
        sequence_output_ec_img = self.dropout(sequence_output_ec_img)
        moe_output_ec_img, balance_loss_ec_img, _ = self.moe(sequence_output_ec_img, input_ids_img, input_mask_img)
        
        
        features_ec_text = self.roberta(input_ids_text, attention_mask=input_mask_text)
        sequence_output_ec_text = features_ec_text["last_hidden_state"]
        sequence_output_ec_text = self.dropout(sequence_output_ec_text)
        moe_output_ec_text, balance_loss_ec_text, _ = self.moe(sequence_output_ec_text, input_ids_text, input_mask_text)
        
        # # Independent probability calculation for each token
        probabilities_text = F.softmax(self.classifier(moe_output_ec_text), dim=-1)  # PθT(yi|x, I)
        probabilities_img = F.softmax(self.classifier(moe_output_ec_img), dim=-1)    # PθI(yi|x, I)

        # Reduce p_T to a scalar per batch instance
        p_T_scalar = torch.mean(p_T, dim=-1, keepdim=True)  # Shape: [batch_size, 1]
        p_I_scalar = 1 - p_T_scalar  # Shape: [batch_size, 1]

        # Expand dimensions to match probabilities_text and probabilities_img
        p_T_expanded = p_T_scalar.unsqueeze(1).expand(-1, probabilities_text.size(1), -1)  # Shape: [batch_size, seq_len, 1]
        p_I_expanded = p_I_scalar.unsqueeze(1).expand(-1, probabilities_img.size(1), -1)  # Shape: [batch_size, seq_len, 1]
        
        # Weighted summation of probabilities
        combined_probabilities = (
            p_T_expanded * probabilities_text + p_I_expanded * probabilities_img
        )  # Shape: [batch_size, seq_len, num_labels]
        # Decoding with CRF
        if labels_img is not None:
            loss = -self.crf(combined_probabilities, labels_img, mask=input_mask_img.byte(), reduction="mean")
            return loss
        else:
            predictions = self.crf.decode(combined_probabilities, mask=input_mask_img)
            return predictions

if __name__ == "__main__":
    model = Roberta_token_classification.from_pretrained(r'vinai/phobert-base-v2', cache_dir="cache")
    # print(model)

    import pickle
    # Load the parameters from the pickle file
    with open("params.pkl", "rb") as file:
        loaded_params = pickle.load(file)

    device = "cpu"
    # Unpack the parameters to feed into the function
    input_ids_text = loaded_params["input_ids_text"].to(device)
    segment_ids_text = loaded_params["segment_ids_text"].to(device)
    input_mask_text = loaded_params["input_mask_text"].to(device)
    input_ids_img = loaded_params["input_ids_img"].to(device)
    segment_ids_img = loaded_params["segment_ids_img"].to(device)
    input_mask_img = loaded_params["input_mask_img"].to(device)
    input_ids_origin = loaded_params["input_ids_origin"].to(device)
    segment_ids_origin = loaded_params["segment_ids_origin"].to(device)
    input_mask_origin = loaded_params["input_mask_origin"].to(device)
    image_features = loaded_params["image_features"].to(device)
    labels_text = loaded_params["labels_text"].to(device)
    labels_img = loaded_params["labels_img"].to(device)
    labels_origin = loaded_params["labels_origin"].to(device)
    
    # Use the loaded parameters to call the model
    neg_log_likelihood = model(
        input_ids_text, segment_ids_text, input_mask_text,
        input_ids_img, segment_ids_img, input_mask_img,
        input_ids_origin, segment_ids_origin, input_mask_origin,
        image_features, labels_text, labels_img, labels_origin
    )
    print(neg_log_likelihood)