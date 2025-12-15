import numpy as np
import torch
import jittor as jt
def run_test():
    from layers_jt.masking import TriangularCausalMask as TriangularCausalMaskJittor, ProbMask as ProbMaskJittor
    from layers.masking import TriangularCausalMask as TriangularCausalMaskTorch, ProbMask as ProbMaskTorch

    print("=== 开始对比测试 PyTorch vs Jittor ===")
    
    # 设置参数
    B = 2   # Batch size
    H = 4   # Heads
    L = 10  # Full Length
    Q_Len = 5 # Query Length / Sampled Length
    S = 10  # Score last dim (usually equals L)
    
    device = "cpu" # 测试使用 CPU，方便对比

    # ---------------------------
    # 测试 1: TriangularCausalMask
    # ---------------------------
    print("\n[Test 1] TriangularCausalMask...")
    
    # PyTorch 运行
    torch_mask = TriangularCausalMaskTorch(B, L, device).mask
    torch_res = torch_mask.numpy()
    
    # Jittor 运行
    jittor_mask = TriangularCausalMaskJittor(B, L, device).mask
    jittor_res = jittor_mask.numpy() # Jittor需要显式调用.numpy()同步数据
    
    # 对比
    if np.array_equal(torch_res, jittor_res):
        print("✅ TriangularCausalMask 结果一致！")
    else:
        print("❌ TriangularCausalMask 结果不一致！")
        print("Diff:", np.abs(torch_res.astype(int) - jittor_res.astype(int)).sum())

    # ---------------------------
    # 测试 2: ProbMask
    # ---------------------------
    print("\n[Test 2] ProbMask...")
    
    # 构造模拟数据
    # index: 代表在 ProbSparse Attention 中选出的 Query 的索引
    # 形状通常是 [B, H, Q_Len]，值在 [0, L) 之间
    np_index = np.random.randint(0, L, size=(B, H, Q_Len))
    
    # scores: 用作形状参考的 tensor
    # 形状 [B, H, Q_Len, S]
    np_scores = np.random.randn(B, H, Q_Len, S)
    
    # 准备 PyTorch 输入
    t_index = torch.from_numpy(np_index).long().to(device)
    t_scores = torch.from_numpy(np_scores).to(device)
    
    # 准备 Jittor 输入
    j_index = jt.array(np_index)
    j_scores = jt.array(np_scores)
    
    # PyTorch 运行
    pm_torch = ProbMaskTorch(B, H, L, t_index, t_scores, device)
    torch_res = pm_torch.mask.cpu().numpy()
    
    # Jittor 运行
    pm_jittor = ProbMaskJittor(B, H, L, j_index, j_scores, device)
    jittor_res = pm_jittor.mask.numpy()
    
    # 对比
    # 注意：Jittor 的 bool 类型转 numpy 也是 bool，可以直接对比
    if np.array_equal(torch_res, jittor_res):
        print("✅ ProbMask 结果一致！")
    else:
        print("❌ ProbMask 结果不一致！")
        # 调试信息
        print(f"Shape Torch: {torch_res.shape}")
        print(f"Shape Jittor: {jittor_res.shape}")
        diff = np.abs(torch_res.astype(int) - jittor_res.astype(int))
        print(f"差异元素数量: {diff.sum()}")

def test_attention_correctness():
    # 原始 PyTorch 类定义 (为了测试需要在这里临时定义，或者从你的文件中导入)
    # 此处假设 TriangularCausalMaskTorch, FullAttentionTorch, AttentionLayerTorch 
    # 是你问题中提供的原始 PyTorch 代码
    from layers.SelfAttention_Family import AttentionLayer as AttentionLayerTorch
    from layers.SelfAttention_Family import FullAttention as FullAttentionTorch
    from layers_jt.SelfAttention_Family import FullAttention, AttentionLayer
    print("=== 开始对比 AttentionLayer PyTorch vs Jittor ===")
    
    # 1. 设置参数
    B, L, S = 2, 10, 10
    d_model = 32
    n_heads = 4
    
    # 2. 构造输入数据 (固定随机种子)
    np.random.seed(0)
    np_queries = np.random.randn(B, L, d_model).astype(np.float32)
    np_keys = np.random.randn(B, S, d_model).astype(np.float32)
    np_values = np.random.randn(B, S, d_model).astype(np.float32)
    
    # 3. 初始化 PyTorch 模型
    # 必须 eval() 模式，因为 Dropout 随机性会导致无法对比
    torch_attn = FullAttentionTorch(mask_flag=True, factor=5, scale=None, attention_dropout=0.0, output_attention=True)
    torch_layer = AttentionLayerTorch(torch_attn, d_model, n_heads)
    torch_layer.eval()
    
    # 4. 初始化 Jittor 模型
    jittor_attn = FullAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.0, output_attention=True)
    jittor_layer = AttentionLayer(jittor_attn, d_model, n_heads)
    jittor_layer.eval()
    
    # 5. *** 关键步骤: 权重同步 ***
    # 将 PyTorch 的权重复制给 Jittor，确保计算起点一致
    # PyTorch Linear weight: [out_features, in_features]
    # Jittor Linear weight: [out_features, in_features]
    # Bias: [out_features]
    # 两者内存布局一致，直接赋值即可
    
    def copy_weights(jt_layer, torch_layer):
        # Query Projection
        jt_layer.query_projection.weight.assign(torch_layer.query_projection.weight.detach().numpy())
        jt_layer.query_projection.bias.assign(torch_layer.query_projection.bias.detach().numpy())
        # Key Projection
        jt_layer.key_projection.weight.assign(torch_layer.key_projection.weight.detach().numpy())
        jt_layer.key_projection.bias.assign(torch_layer.key_projection.bias.detach().numpy())
        # Value Projection
        jt_layer.value_projection.weight.assign(torch_layer.value_projection.weight.detach().numpy())
        jt_layer.value_projection.bias.assign(torch_layer.value_projection.bias.detach().numpy())
        # Out Projection
        jt_layer.out_projection.weight.assign(torch_layer.out_projection.weight.detach().numpy())
        jt_layer.out_projection.bias.assign(torch_layer.out_projection.bias.detach().numpy())

    copy_weights(jittor_layer, torch_layer)
    
    # 7. 运行前向传播
    # PyTorch
    t_q = torch.from_numpy(np_queries)
    t_k = torch.from_numpy(np_keys)
    t_v = torch.from_numpy(np_values)
    with torch.no_grad():
        t_out, t_attn_score = torch_layer(t_q, t_k, t_v,None)
    
    # Jittor
    j_q = jt.array(np_queries)
    j_k = jt.array(np_keys)
    j_v = jt.array(np_values)
    j_out, j_attn_score = jittor_layer(j_q, j_k, j_v,None)
    
    # 8. 对比结果
    t_out_np = t_out.numpy()
    j_out_np = j_out.numpy()
    
    t_score_np = t_attn_score.numpy()
    j_score_np = j_attn_score.numpy()
    
    # 检查输出 Feature
    if np.allclose(t_out_np, j_out_np, atol=1e-5):
        print("✅ Output Features 一致!")
    else:
        print("❌ Output Features 不一致!")
        diff = np.abs(t_out_np - j_out_np).max()
        print(f"Max Diff: {diff}")
        
    # 检查 Attention Score
    if np.allclose(t_score_np, j_score_np, atol=1e-5):
        print("✅ Attention Scores 一致!")
    else:
        print("❌ Attention Scores 不一致!")
        diff = np.abs(t_score_np - j_score_np).max()
        print(f"Max Diff: {diff}")
    

if __name__ == "__main__":
    run_test()
    # 假设你把 pytorch代码保存为了 torch_impl.py
    # 这里的运行需要你确保环境中同时安装了 torch 和 jittor
    try:
        test_attention_correctness()
    except ImportError as e:
        print(f"测试需要同时安装 PyTorch 和 Jittor。错误: {e}")
    except Exception as e:
        print(f"运行出错 (可能是因为缺少torch_impl文件，请将原代码保存): {e}")
    