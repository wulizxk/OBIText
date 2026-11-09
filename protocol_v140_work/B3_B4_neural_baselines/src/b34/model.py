from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class BiLSTMMLM(nn.Module):
    def __init__(self, vocab_size, target_size, embedding_dim, hidden_size, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, parameter in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                quarter = parameter.numel() // 4
                parameter.data[quarter : 2 * quarter].fill_(1.0)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def selected_logits(self, input_ids, attention_mask, owners, positions):
        lengths = attention_mask.sum(dim=1).to("cpu")
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=input_ids.shape[1]
        )
        selected = hidden[owners, positions]
        return self.classifier(self.dropout(self.norm(selected)))

    def model_spec(self):
        return {
            "model_family": "bilstm_mlm",
            "vocab_size": self.embedding.num_embeddings,
            "target_size": self.classifier.out_features,
            "embedding_dim": self.embedding.embedding_dim,
            "hidden_size": self.encoder.hidden_size,
            "layers": self.encoder.num_layers,
            "dropout": self.dropout.p,
        }


def autocast(device, cfg):
    if device.type == "cuda" and cfg["precision"] == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_model(codec, cfg):
    if cfg["model_family"] == "bilstm_mlm":
        return BiLSTMMLM(
            codec.vocab_size,
            len(codec.units),
            cfg["input_embedding_dim"],
            cfg["lstm_hidden_size_per_direction"],
            cfg["lstm_layers"],
            cfg["dropout"],
        )
    if cfg["model_family"] == "bert_mlm_random_init":
        from transformers import BertConfig, BertForMaskedLM

        config = BertConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            hidden_act=cfg["hidden_act"],
            hidden_dropout_prob=cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=cfg["attention_probs_dropout_prob"],
            max_position_embeddings=cfg["max_position_embeddings"],
            type_vocab_size=cfg["type_vocab_size"],
            initializer_range=cfg["initializer_range"],
            layer_norm_eps=cfg["layer_norm_eps"],
            pad_token_id=codec.pad_id,
        )
        config._attn_implementation = "eager"
        model = BertForMaskedLM(config)
        model.tie_weights()
        return model
    raise ValueError("Unknown model family")


def model_spec(model, cfg):
    if cfg["model_family"] == "bilstm_mlm":
        return model.model_spec()
    return {"model_family": "bert_mlm_random_init", "bert_config": model.config.to_dict()}


def model_from_spec(spec):
    if spec["model_family"] == "bilstm_mlm":
        return BiLSTMMLM(
            spec["vocab_size"],
            spec["target_size"],
            spec["embedding_dim"],
            spec["hidden_size"],
            spec["layers"],
            spec["dropout"],
        )
    if spec["model_family"] == "bert_mlm_random_init":
        from transformers import BertConfig, BertForMaskedLM

        config = BertConfig.from_dict(spec["bert_config"])
        config._attn_implementation = "eager"
        model = BertForMaskedLM(config)
        model.tie_weights()
        return model
    raise ValueError("Unknown checkpoint model family")


def restricted_logits(model, cfg, input_ids, attention_mask, owners, positions, target_ids):
    if cfg["model_family"] == "bilstm_mlm":
        return model.selected_logits(input_ids, attention_mask, owners, positions)
    hidden = model.bert(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    selected = hidden[owners, positions]
    transformed = model.cls.predictions.transform(selected)
    decoder = model.get_output_embeddings()
    return F.linear(
        transformed,
        decoder.weight.index_select(0, target_ids),
        decoder.bias.index_select(0, target_ids),
    )


def per_sample_losses(model, cfg, batch, target_ids):
    logits = restricted_logits(
        model,
        cfg,
        batch["input_ids"],
        batch["attention_mask"],
        batch["owners"],
        batch["positions"],
        target_ids,
    )
    token_loss = F.cross_entropy(logits.float(), batch["targets"], reduction="none")
    totals = torch.zeros(len(batch["counts"]), device=logits.device).scatter_add(
        0, batch["owners"], token_loss
    )
    return totals / batch["counts"]


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        lower = name.lower()
        target = no_decay if name.endswith("bias") or "norm" in lower else decay
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=tuple(cfg["adam_betas"]),
        eps=cfg["adam_epsilon"],
        foreach=False,
    )
