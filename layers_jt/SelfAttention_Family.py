import jittor as jt
from jittor import nn
from math import sqrt
from layers_jt.masking import TriangularCausalMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def execute(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Jittor 中通常习惯用 execute 替代 forward，但 forward 也可以
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # PyTorch: scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # 逻辑: (B, L, H, E) x (B, S, H, E)^T -> (B, H, L, S)
        # Jittor 实现: 显式转置后矩阵乘法
        queries_opt = queries.permute(0, 2, 1, 3) # [B, H, L, E]
        keys_opt = keys.permute(0, 2, 3, 1)       # [B, H, E, S]
        scores = jt.matmul(queries_opt, keys_opt) # [B, H, L, S]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            # PyTorch: scores.masked_fill_(attn_mask.mask, -np.inf)
            # Jittor: 建议使用 where 非原地操作，且使用一个极小值替代 -inf 以避免梯度计算出现 NaN
            # 虽然 -np.inf 在前向传播没问题，但在某些优化器下可能会有问题，这里保持逻辑一致使用 -1e9 或 -inf
            scores = jt.where(attn_mask.mask, -1e9, scores)

        # Softmax & Dropout
        A = self.dropout(jt.nn.softmax(scale * scores, dim=-1))

        # PyTorch: V = torch.einsum("bhls,bshd->blhd", A, values)
        # 逻辑: (B, H, L, S) x (B, S, H, D) -> (B, H, L, D) -> permute to (B, L, H, D)
        values_opt = values.permute(0, 2, 1, 3) # [B, H, S, D]
        V = jt.matmul(A, values_opt)            # [B, H, L, D]
        V = V.permute(0, 2, 1, 3)               # [B, L, H, D]

        if self.output_attention:
            return V, A
        else:
            return V, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def execute(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Jittor 的 Linear 行为与 PyTorch 一致，直接调用
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        
        # 展平
        out = out.view(B, L, -1)

        return self.out_projection(out), attn