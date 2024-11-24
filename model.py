from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, head_size: int, n_embds: int, dropout: float):
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(n_embds, head_size, bias=False)
        self.query = nn.Linear(n_embds, head_size, bias=False)
        self.value = nn.Linear(n_embds, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, n_embds = x.shape

        # Compute Q, K, V matrices
        q = self.query(x)  # (B,T,head_size)
        k = self.key(x)    # (B,T,head_size)
        v = self.value(x)  # (B,T,head_size)

        # Compute scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B,T,T)

        # Apply attention mask to ignore padding tokens
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T) to broadcast along attention weights
            wei = wei.masked_fill(mask == 0, float('-inf'))  # Set -inf where mask is 0 (padding)

        wei = F.softmax(wei, dim=-1)  # Softmax on the last dimension (attention scores)
        wei = self.dropout(wei)       # Apply dropout

        # Weighted sum of values
        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.attn_heads = nn.ModuleList([SelfAttention(head_size, head_size * num_heads, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, head_size * num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Each head processes the input independently and then we concatenate
        multi_head_out = torch.cat([head(x, mask) for head in self.attn_heads], dim=-1)  # (B, T, n_embds)
        out = self.dropout(self.proj(multi_head_out))  # Project concatenated heads back to n_embds
        return out


class FeedForwardLayer(nn.Module):

    def __init__(self, n_embds: int, dropout: float) -> None:
        super().__init__()
        self.ffw = nn.Sequential(
            nn.Linear(n_embds, n_embds * 4),
            nn.ReLU(),
            nn.Linear(n_embds * 4, n_embds),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.ffw(x)
        return out


class Block(nn.Module):

    def __init__(self, n_embds: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embds // num_heads
        self.multi_headed_attn = MultiHeadSelfAttention(num_heads, head_size, dropout)
        self.ffw_layer = FeedForwardLayer(n_embds, dropout)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, x, mask=None):
        # Apply multi-head attention and add residual connection
        x = x + self.multi_headed_attn(self.ln1(x), mask)
        # Apply feed-forward network and add residual connection
        x = x + self.ffw_layer(self.ln2(x))
        return x


class GPTDecoder(nn.Module):

    def __init__(self, model_params: Dict) -> None:
        super().__init__()

        self._initialise_model_params(model_params)
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.n_embds)
        self.positional_embeddings = nn.Embedding(self.block_size, self.n_embds)
        self.blocks = nn.ModuleList([Block(self.n_embds, self.n_heads, self.dropout) for _ in range(self.n_layers)])
        self.layernorm = nn.LayerNorm(self.n_embds)
        self.classifier = nn.Linear(self.n_embds, self.num_classes)

    def _initialise_model_params(self, model_params: Dict) -> None:

        self.vocab_size = model_params.get("vocab_size")
        self.n_embds = model_params.get("num_embeddings")
        self.block_size = model_params.get("block_size")
        self.n_heads = model_params.get("num_heads")
        self.n_layers = model_params.get("num_layers")
        self.num_classes = model_params.get("output_classes")
        self.dropout = model_params.get("dropout")
        self.device = model_params.get("device")

    def forward(self, idx: torch.tensor, targets=None, mask: torch.tensor = None) -> Tuple[torch.tensor, float]:

        B, T = idx.shape
        token_embeddings = self.token_embeddings(idx)  # B, T -> B, T, n_embds
        positional_embeddings = self.positional_embeddings(torch.arange(T, device = self.device))  # T, n_embds

        # Combine token and positional embeddings
        x = token_embeddings + positional_embeddings  # B, T, n_embds

        # Forward pass through each Block, passing the mask
        for block in self.blocks:
            x = block(x, mask)  # Pass mask to each block individually

        layer_norm_out = self.layernorm(x)  # (B, T, n_embds)

        # Pooling to get a single vector per sequence for classification
        pooled_output = layer_norm_out.mean(dim=1)  # Average pooling across T dimension

        logits = self.classifier(pooled_output)  # B, n_embds -> B, num_classes

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)  # Cross-entropy loss for classification

        return logits, loss
