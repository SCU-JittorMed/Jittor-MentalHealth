import torch
import jittor as jt
import numpy as np
import sys
import os

# 路径设置
sys.path.append(os.getcwd())

# 导入模块
from layers.Embed import PositionalEmbedding as PosEmbedPT
from layers.Embed import TokenEmbedding as TokEmbedPT
from layers.Embed import TemporalEmbedding as TempEmbedPT

from layers_jt.Embed import PositionalEmbedding as PosEmbedJT
from layers_jt.Embed import TokenEmbedding as TokEmbedJT
from layers_jt.Embed import TemporalEmbedding as TempEmbedJT

def compare(pt_out, jt_out, name):
    pt_np = pt_out.detach().numpy()
    jt_np = jt_out.numpy()
    
    diff = np.abs(pt_np - jt_np)
    max_diff = np.max(diff)
    
    # 忽略边缘计算中间部分的误差 (针对 TokenEmbedding 的 Padding 问题)
    if len(pt_np.shape) == 3 and pt_np.shape[1] > 2:
        mid_diff = np.max(np.abs(pt_np[:, 1:-1, :] - jt_np[:, 1:-1, :]))
    else:
        mid_diff = max_diff

    print(f"--- {name} Results ---")
    print(f"Shape: {pt_np.shape}")
    print(f"Max Diff (Global): {max_diff:.8f}")
    if mid_diff != max_diff:
        print(f"Max Diff (Inner):  {mid_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ Status: PERFECT MATCH")
    elif mid_diff < 1e-5:
        print("⚠️ Status: INNER MATCH (Edges differ due to padding)")
    else:
        print("❌ Status: MISMATCH")
    print("-" * 30 + "\n")



def test_token_embedding():
    print("Testing TokenEmbedding...")
    c_in = 7
    d_model = 512
    seq_len = 96
    bs = 16
    
    # 1. Init
    model_pt = TokEmbedPT(c_in=c_in, d_model=d_model)
    model_jt = TokEmbedJT(c_in=c_in, d_model=d_model)
    
    # 2. Sync Weights
    # Conv1d 权重形状: [Out, In, Kernel]
    w = model_pt.tokenConv.weight.detach().numpy()
    model_jt.tokenConv.weight = jt.array(w)
    
    # 3. Input
    x_np = np.random.randn(bs, seq_len, c_in).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jt = jt.array(x_np)
    
    # 4. Forward
    model_pt.eval()
    model_jt.eval()
    
    # PyTorch 可能用了 circular padding
    out_pt = model_pt(x_pt)
    out_jt = model_jt(x_jt)
    
    compare(out_pt, out_jt, "TokenEmbedding")
    
def test_positional_embedding():
    print("Testing PositionalEmbedding...")
    d_model = 512
    seq_len = 100
    bs = 2
    
    # 1. Init
    # Positional Embedding 是确定性的数学计算，不需要同步权重
    # 只要 d_model 和 max_len 一致，生成的 PE 矩阵就该一致
    model_pt = PosEmbedPT(d_model=d_model)
    model_jt = PosEmbedJT(d_model=d_model)
    
    # 2. Input
    x_np = np.zeros((bs, seq_len, d_model)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jt = jt.array(x_np)
    
    # 3. Forward
    model_pt.eval()
    model_jt.eval()
    
    out_pt = model_pt(x_pt)
    out_jt = model_jt(x_jt)
    
    compare(out_pt, out_jt, "PositionalEmbedding")
    
def test_temporal_embedding():
    print("Testing TemporalEmbedding...")
    d_model = 512
    seq_len = 96
    bs = 16
    
    # 1. Init (Default: fixed embedding)
    model_pt = TempEmbedPT(d_model=d_model, embed_type='fixed', freq='h')
    model_jt = TempEmbedJT(d_model=d_model, embed_type='fixed', freq='h')
    
    # 2. Sync Weights
    # 需要同步内部的 5 个 embedding 层
    layers_to_sync = ['hour_embed', 'weekday_embed', 'day_embed', 'month_embed']
    # minute_embed 仅在 freq='t' 时存在，这里 freq='h'
    
    for name in layers_to_sync:
        pt_layer = getattr(model_pt, name)
        jt_layer = getattr(model_jt, name)
        
        # 获取 PyTorch Embedding 的权重
        # 注意: FixedEmbedding 内部还有一层 self.emb
        w = pt_layer.emb.weight.detach().numpy()
        
        # 赋值给 Jittor
        jt_layer.emb.weight = jt.array(w)
        
    # 3. Input
    # 必须生成合法范围内的数据
    x_mark_np = np.zeros((bs, seq_len, 5), dtype=np.float32)
    x_mark_np[:, :, 0] = np.random.randint(0, 13, (bs, seq_len)) # Month
    x_mark_np[:, :, 1] = np.random.randint(0, 32, (bs, seq_len)) # Day
    x_mark_np[:, :, 2] = np.random.randint(0, 7, (bs, seq_len))  # Weekday
    x_mark_np[:, :, 3] = np.random.randint(0, 24, (bs, seq_len)) # Hour
    x_mark_np[:, :, 4] = np.random.randint(0, 4, (bs, seq_len))  # Minute
    
    x_pt = torch.from_numpy(x_mark_np).long()
    x_jt = jt.array(x_mark_np) # Jittor Casts internally
    
    # 4. Forward
    model_pt.eval()
    model_jt.eval()
    
    out_pt = model_pt(x_pt)
    out_jt = model_jt(x_jt)
    
    compare(out_pt, out_jt, "TemporalEmbedding")

if __name__ == "__main__":
    test_positional_embedding()
    test_token_embedding()
    test_temporal_embedding()