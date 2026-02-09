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


def asym_adj(adj):
    adj = np.asarray(adj, dtype=np.float32)
    rowsum = adj.sum(axis=1)
    d_inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    d_inv[nonzero] = 1.0 / rowsum[nonzero]
    return (adj.T * d_inv).T


def compute_support_gwn(adj):
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    adj = np.asarray(adj, dtype=np.float32)
    return [
        torch.tensor(asym_adj(adj), dtype=torch.float32),
        torch.tensor(asym_adj(adj.T), dtype=torch.float32),
    ]


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


class DiffusionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, diffusion_steps=2, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diffusion_steps = diffusion_steps
        self.mlp = nn.Conv2d(
            in_channels * (diffusion_steps * 2 + 1),
            out_channels,
            kernel_size=(1, 1),
            bias=bias,
        )

    def forward(self, x, base_shape, supports):
        B, C, _ = x.shape
        K, L = base_shape[2], base_shape[3]
        x = x.view(B, C, K, L)

        out = [x]

        for support in supports:
            x1 = x
            for _ in range(1, self.diffusion_steps + 1):
                x1 = torch.einsum("ij,bcjl->bcil", support, x1)
                out.append(x1)

        x_cat = torch.cat(out, dim=1)
        x_out = self.mlp(x_cat)
        x_out = x_out.view(B, self.out_channels, K * L)
        return x_out


def _resolve_graph_lags(config):
    if config is None:
        return [0]
    lags = config.get("graph_lags")
    if lags is None:
        max_lag = config.get("graph_max_lag")
        if max_lag is None:
            return [0]
        try:
            max_lag = int(max_lag)
        except (TypeError, ValueError):
            return [0]
        if max_lag <= 0:
            return [0]
        return list(range(max_lag + 1))
    if isinstance(lags, int):
        if lags <= 0:
            return [0]
        return list(range(lags + 1))
    if isinstance(lags, (list, tuple)):
        cleaned = []
        for lag in lags:
            try:
                cleaned.append(int(lag))
            except (TypeError, ValueError):
                continue
        cleaned = sorted({lag for lag in cleaned if lag >= 0})
        return cleaned or [0]
    return [0]


class LaggedDiffusionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, diffusion_steps=2, lags=None, learnable=True):
        super().__init__()
        self.lags = lags if lags is not None else [0]
        self.graph_conv = DiffusionGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            diffusion_steps=diffusion_steps,
        )
        self.learnable = learnable and len(self.lags) > 1
        if self.learnable:
            self.lag_logits = nn.Parameter(torch.zeros(len(self.lags)))
        else:
            self.register_buffer("lag_logits", torch.zeros(len(self.lags)))

    def _shift_time(self, x, base_shape, lag):
        if lag == 0:
            return x
        B, C, K, L = base_shape
        x = x.view(B, C, K, L)
        if lag >= L:
            return torch.zeros_like(x).view(B, C, K * L)
        out = torch.zeros_like(x)
        out[..., lag:] = x[..., : L - lag]
        return out.view(B, C, K * L)

    def forward(self, x, base_shape, supports):
        if len(self.lags) == 1 and self.lags[0] == 0:
            return self.graph_conv(x, base_shape, supports)
        outputs = []
        for lag in self.lags:
            x_lag = self._shift_time(x, base_shape, lag)
            outputs.append(self.graph_conv(x_lag, base_shape, supports))
        stacked = torch.stack(outputs, dim=0)
        weights = torch.softmax(self.lag_logits, dim=0).view(-1, 1, 1, 1)
        return (weights * stacked).sum(dim=0)


class GraphormerSpatialAttention(nn.Module):
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


class GraphDiffusionSpatialLayer(nn.Module):
    def __init__(self, channels, diffusion_steps, supports):
        super().__init__()
        if supports is None or len(supports) < 2:
            raise ValueError("GraphDiffusionSpatialLayer requires graph supports.")
        self.graph_conv = DiffusionGraphConv(
            in_channels=channels,
            out_channels=channels,
            diffusion_steps=diffusion_steps,
        )
        self.register_buffer("support_0", supports[0])
        self.register_buffer("support_1", supports[1])

    def forward(self, x, base_shape):
        supports = [self.support_0, self.support_1]
        return self.graph_conv(x, base_shape, supports)


class DiffGraphAwareConditionalDiffusion(nn.Module):
    def __init__(self, config, inputdim=2, adj=None):
        super().__init__()
        if adj is None:
            raise ValueError("DiffGraphAwareConditionalDiffusion requires an adjacency matrix.")
        self.channels = config["channels"]
        self.graph_model = config.get("graph_model", "graphormer")
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        supports = None
        graph_max_hops = config.get("graph_max_hops", 6)
        graph_dropout = config.get("graph_dropout", 0.1)
        graph_ff_dim = config.get("graph_ff_dim", 64)
        graph_diffusion_steps = config.get("graph_diffusion_steps", 2)

        if self.graph_model == "graphdiffusion":
            supports = compute_support_gwn(adj)
        elif self.graph_model != "graphormer":
            raise ValueError(f"Unknown graph_model: {self.graph_model}")

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config.get("is_linear", False),
                    graph_model=self.graph_model,
                    adj=adj,
                    supports=supports,
                    graph_max_hops=graph_max_hops,
                    graph_dropout=graph_dropout,
                    graph_ff_dim=graph_ff_dim,
                    graph_diffusion_steps=graph_diffusion_steps,
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
        supports=None,
        graph_max_hops=6,
        graph_dropout=0.1,
        graph_ff_dim=64,
        graph_diffusion_steps=2,
        is_linear=False,
        graph_model="graphormer",
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
        self.graph_model = graph_model
        if self.graph_model == "graphormer":
            self.graph_layer = GraphormerSpatialAttention(
                channels=channels,
                nheads=nheads,
                adj=adj,
                max_hops=graph_max_hops,
                dropout=graph_dropout,
                dim_feedforward=graph_ff_dim,
            )
        elif self.graph_model == "graphdiffusion":
            self.graph_layer = GraphDiffusionSpatialLayer(
                channels=channels,
                diffusion_steps=graph_diffusion_steps,
                supports=supports,
            )
        else:
            raise ValueError(f"Unknown graph_model: {self.graph_model}")

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

        if self.graph_model == "graphdiffusion":
            y = self.forward_feature(y, base_shape)
            y = self.forward_time(y, base_shape)
        else:
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
