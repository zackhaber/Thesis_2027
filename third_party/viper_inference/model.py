"""Copied from third_party/ViPER/model.py; training-only QNet omitted."""

import math

import torch
from torch import nn


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.key_dim = embedding_dim
        self.norm_factor = 1 / math.sqrt(embedding_dim)
        self.tanh_clipping = 10
        self.w_query = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.w_key = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)

    def forward(self, query, keys, mask=None):
        batch, key_count, dimensions = keys.size()
        query_count = query.size(1)
        key = torch.matmul(keys.reshape(-1, dimensions), self.w_key).view(batch, key_count, -1)
        q = torch.matmul(query.reshape(-1, dimensions), self.w_query).view(batch, query_count, -1)
        scores = self.norm_factor * torch.matmul(q, key.transpose(1, 2))
        scores = self.tanh_clipping * torch.tanh(scores)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e8)
        return torch.log_softmax(scores, dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim // n_heads
        self.norm_factor = 1 / math.sqrt(self.value_dim)
        self.w_query = nn.Parameter(torch.Tensor(n_heads, embedding_dim, self.value_dim))
        self.w_key = nn.Parameter(torch.Tensor(n_heads, embedding_dim, self.value_dim))
        self.w_value = nn.Parameter(torch.Tensor(n_heads, embedding_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(n_heads, self.value_dim, embedding_dim))
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        k = q if k is None else k
        v = q if v is None else v
        batch, key_count, dimensions = k.size()
        query_count = q.size(1)
        value_count = v.size(1)
        key = torch.matmul(k.reshape(-1, dimensions), self.w_key).view(self.n_heads, batch, key_count, -1)
        value = torch.matmul(v.reshape(-1, dimensions), self.w_value).view(self.n_heads, batch, value_count, -1)
        query = torch.matmul(q.reshape(-1, dimensions), self.w_query).view(self.n_heads, batch, query_count, -1)
        scores = self.norm_factor * torch.matmul(query, key.transpose(2, 3))
        if attn_mask is not None:
            mask = attn_mask.view(1, batch, query_count, key_count).expand_as(scores)
            scores = scores.masked_fill(mask > 0, -1e8)
        if key_padding_mask is not None:
            mask = key_padding_mask.repeat(1, query_count, 1)
            mask = mask.view(1, batch, query_count, key_count).expand_as(scores)
            scores = scores.masked_fill(mask > 0, -1e8)
        attention = torch.softmax(scores, dim=-1)
        heads = torch.matmul(attention, value)
        return torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.reshape(-1, self.embedding_dim),
        ).view(-1, query_count, self.embedding_dim), attention


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, value):
        return self.normalizer(value.reshape(-1, value.size(-1))).view(*value.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        initial = src
        hidden = self.normalization1(src)
        hidden, _ = self.multiHeadAttention(hidden, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        hidden = hidden + initial
        return self.feedForward(self.normalization2(hidden)) + hidden


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, target, memory, key_padding_mask=None, attn_mask=None):
        initial = target
        target = self.normalization1(target)
        memory = self.normalization1(memory)
        target, weights = self.multiHeadAttention(target, memory, memory,
                                                  key_padding_mask=key_padding_mask,
                                                  attn_mask=attn_mask)
        target = target + initial
        return self.feedForward(self.normalization2(target)) + target, weights


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for _ in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask, attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(embedding_dim, n_head) for _ in range(n_layer))

    def forward(self, target, memory, key_padding_mask=None, attn_mask=None):
        weights = None
        for layer in self.layers:
            target, weights = layer(target, memory, key_padding_mask, attn_mask)
        return target, weights


class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super().__init__()
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.graph_encoder = Encoder(embedding_dim, 4, 6)
        self.graph_decoder = Decoder(embedding_dim, 4, 1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.pointer = SingleHeadAttention(embedding_dim)

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        embedding = self.initial_embedding(node_inputs)
        embedding = self.graph_encoder(embedding, node_padding_mask, edge_mask)
        dimensions = embedding.size(2)
        current = torch.gather(embedding, 1, current_index.repeat(1, 1, dimensions))
        decoded, _ = self.graph_decoder(current, embedding, node_padding_mask)
        state = self.current_embedding(torch.cat((decoded, current), dim=-1))
        neighbors = torch.gather(embedding, 1, current_edge.repeat(1, 1, dimensions))
        return self.pointer(state, neighbors, edge_padding_mask).squeeze(1)