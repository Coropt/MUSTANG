import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=localheads,
        local_attn_window_size=localwindow,
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _shortest_path_hops(adj):
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square matrix for hop distance computation.")
    k = adj.shape[0]
    edges = adj != 0
    dist = np.full((k, k), k + 1, dtype=np.int64)
    for i in range(k):
        dist[i, i] = 0
        queue = deque([i])
        while queue:
            v = queue.popleft()
            next_dist = dist[i, v] + 1
            for u in np.where(edges[v])[0]:
                if dist[i, u] > next_dist:
                    dist[i, u] = next_dist
                    queue.append(u)
    return dist


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class GraphAwareSpatialAttention(nn.Module):
    """
    Graph-aware conditional diffusion style spatial attention with per-head bias.
    
    Key features:
    - Per-head spatial bias (each head learns independent bias)
    - Normal initialization for bias embedding (std=0.02)
    - Post-LN (aligned with conditional diffusion TransformerEncoderLayer style)
    """
    def __init__(
        self,
        channels,
        nheads,
        adj,
        max_hops=6,
        dropout=0.1,
        dim_feedforward=64,
    ):
        super().__init__()
        self.nheads = nheads
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=nheads, dropout=dropout, batch_first=True
        )
        max_hops = int(max_hops) if max_hops is not None else 0
        max_hops = max(max_hops, 0)
        hop_dist = _shortest_path_hops(adj)
        hop_dist = np.where(hop_dist > max_hops, max_hops + 1, hop_dist)
        self.register_buffer("hop_index", torch.tensor(hop_dist, dtype=torch.long))
        # Per-head bias: each head learns independent spatial bias
        self.bias_embedding = nn.Embedding(max_hops + 2, nheads)
        # Normal initialization with std=0.02
        nn.init.normal_(self.bias_embedding.weight, mean=0.0, std=0.02)

        self.linear1 = nn.Linear(channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, base_shape):
        B, C, K, L = base_shape
        batch_size = B * L
        x = x.reshape(B, C, K, L).permute(0, 3, 2, 1).reshape(batch_size, K, C)
        
        # Per-head bias: (K, K) -> (K, K, nheads) -> (nheads, K, K)
        bias = self.bias_embedding(self.hop_index)  # (K, K, nheads)
        bias = bias.permute(2, 0, 1)  # (nheads, K, K)
        bias = bias.to(dtype=x.dtype, device=x.device)
        # Expand bias to match PyTorch MHA expected shape: (batch_size * nheads, K, K)
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, nheads, K, K)
        bias = bias.reshape(batch_size * self.nheads, K, K)  # (batch_size * nheads, K, K)
        
        # Self-attention with Post-LN (aligned with conditional diffusion style)
        out, _ = self.attn(x, x, x, attn_mask=bias)
        x = x + self.dropout1(out)
        x = self.norm1(x)
        
        # FFN with Post-LN (aligned with conditional diffusion style)
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        
        x = x.reshape(B, L, K, C).permute(0, 3, 2, 1).reshape(B, C, K * L)
        return x


class DiffGraphAwareConditionalDiffusion(nn.Module):
    def __init__(self, config, inputdim=2, adj=None):
        super().__init__()
        if adj is None:
            raise ValueError("DiffGraphAwareConditionalDiffusion requires an adjacency matrix.")
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        graph_max_hops = config.get("graph_max_hops", 6)
        graph_dropout = config.get("graph_dropout", 0.1)
        graph_ff_dim = config.get("graph_ff_dim", 64)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config.get("is_linear", False),
                    adj=adj,
                    graph_max_hops=graph_max_hops,
                    graph_dropout=graph_dropout,
                    graph_ff_dim=graph_ff_dim,
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        side_dim,
        channels,
        diffusion_embedding_dim,
        nheads,
        adj,
        graph_max_hops=6,
        graph_dropout=0.1,
        graph_ff_dim=64,
        is_linear=False,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.graph_layer = GraphAwareSpatialAttention(
            channels=channels,
            nheads=nheads,
            adj=adj,
            max_hops=graph_max_hops,
            dropout=graph_dropout,
            dim_feedforward=graph_ff_dim,
        )

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        return self.graph_layer(y, base_shape)

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# Backward-compatible alias
diff_CSDI_GCN = DiffGraphAwareConditionalDiffusion
