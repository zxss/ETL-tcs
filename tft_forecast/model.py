"""
Temporal Fusion Transformer (компактная, но честная реализация).

Реализованы ключевые блоки оригинальной архитектуры (Lim et al., 2021):
  * Gated Linear Unit (GLU) + Gated Residual Network (GRN)
  * Variable Selection Network (VSN) для time-varying входов
  * Static covariate encoders: embedding тикера → контекстные векторы
  * LSTM-энкодер последовательности с static-инициализацией
  * Static enrichment + Interpretable Multi-Head Attention
  * Квантильные головы (pinball loss) для всех целей features.TARGET_COLS:
    дневные (next_low/high, overnight/intraday/total) и недельные
    (week_low/high/total за WEEK_H дней) — единый multi-horizon прогноз

Модель ОДНА на все тикеры: кросс-секционные паттерны рынка извлекаются
общими весами, а специфика инструмента — через static-embedding тикера.

Импортируется только когда доступен torch (см. forecast.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Сетка квантилей: края для коридора/риска + промежуточные для интерполяции
# вероятности прибыли (ProbProfit) по предсказанной квантильной функции.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network с опциональным контекстом."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 context_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ctx = nn.Linear(context_dim, hidden, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(hidden)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        self.out_fc = nn.Linear(hidden, out_dim) if hidden != out_dim else None
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, context=None):
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = torch.relu(h)
        h = self.fc2(h)
        h = self.drop(h)
        h = self.glu(h)
        if self.out_fc is not None:
            h = self.out_fc(h)
        skip = x if self.proj is None else self.proj(x)
        return self.norm(skip + h)


class VariableSelection(nn.Module):
    """VSN: per-feature GRN + softmax-веса значимости признаков."""

    def __init__(self, n_features: int, hidden: int, context_dim: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        # каждый скалярный признак → hidden через свой GRN
        self.feat_grns = nn.ModuleList(
            [GRN(1, hidden, hidden, dropout=dropout) for _ in range(n_features)]
        )
        # веса выбора признаков (с контекстом от static)
        self.weight_grn = GRN(n_features, hidden, n_features,
                              context_dim=context_dim, dropout=dropout)

    def forward(self, x, context):
        # x: (B, L, F) ; context: (B, hidden)
        B, L, F = x.shape
        ctx = context.unsqueeze(1).expand(B, L, -1)
        w = self.weight_grn(x, ctx)                 # (B, L, F)
        w = torch.softmax(w, dim=-1).unsqueeze(-1)  # (B, L, F, 1)

        transformed = torch.stack(
            [self.feat_grns[i](x[..., i:i + 1]) for i in range(F)], dim=-2
        )  # (B, L, F, hidden)
        out = (w * transformed).sum(dim=-2)          # (B, L, hidden)
        return out, w.squeeze(-1)


class TFT(nn.Module):
    def __init__(self, n_features: int, n_tickers: int, hidden: int = 32,
                 n_heads: int = 4, dropout: float = 0.1,
                 n_quantiles: int = len(QUANTILES), n_targets: int = 8):
        super().__init__()
        self.hidden = hidden
        self.n_quantiles = n_quantiles
        self.n_targets = n_targets

        emb_dim = hidden
        self.ticker_emb = nn.Embedding(n_tickers, emb_dim)
        # static encoders → контексты: для VSN, для инициализации LSTM (h,c), для enrichment
        self.ctx_vsn = GRN(emb_dim, hidden, hidden, dropout=dropout)
        self.ctx_h = GRN(emb_dim, hidden, hidden, dropout=dropout)
        self.ctx_c = GRN(emb_dim, hidden, hidden, dropout=dropout)
        self.ctx_enrich = GRN(emb_dim, hidden, hidden, dropout=dropout)

        self.vsn = VariableSelection(n_features, hidden, hidden, dropout)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.post_lstm_gate = GLU(hidden)
        self.post_lstm_norm = nn.LayerNorm(hidden)

        self.enrich = GRN(hidden, hidden, hidden, context_dim=hidden, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.attn_gate = GLU(hidden)
        self.attn_norm = nn.LayerNorm(hidden)
        self.pos_wise = GRN(hidden, hidden, hidden, dropout=dropout)
        self.final_norm = nn.LayerNorm(hidden)

        self.head = nn.Linear(hidden, n_targets * n_quantiles)

    def forward(self, x, ticker_idx):
        # x: (B, L, F) ; ticker_idx: (B,)
        emb = self.ticker_emb(ticker_idx)            # (B, hidden)
        c_vsn = self.ctx_vsn(emb)
        c_h = self.ctx_h(emb)
        c_c = self.ctx_c(emb)
        c_en = self.ctx_enrich(emb)

        sel, _w = self.vsn(x, c_vsn)                 # (B, L, hidden)

        h0 = c_h.unsqueeze(0)                        # (1, B, hidden)
        c0 = c_c.unsqueeze(0)
        lstm_out, _ = self.lstm(sel, (h0, c0))       # (B, L, hidden)
        lstm_out = self.post_lstm_norm(self.post_lstm_gate(lstm_out) + sel)

        enr = self.enrich(lstm_out, c_en.unsqueeze(1).expand_as(lstm_out))

        L = enr.size(1)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(enr, enr, enr, attn_mask=causal)
        attn_out = self.attn_norm(self.attn_gate(attn_out) + enr)

        out = self.pos_wise(attn_out)
        out = self.final_norm(out + attn_out)

        last = out[:, -1, :]                          # (B, hidden) — прогноз на t+1
        y = self.head(last)                          # (B, n_targets * n_quantiles)
        return y.view(-1, self.n_targets, self.n_quantiles)


def quantile_loss(pred, target, quantiles=QUANTILES):
    """Pinball loss. pred: (B, n_targets, Q) ; target: (B, n_targets)."""
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    target = target.unsqueeze(-1)                    # (B, n_targets, 1)
    err = target - pred                              # (B, n_targets, Q)
    loss = torch.maximum(q * err, (q - 1.0) * err)
    return loss.mean()
