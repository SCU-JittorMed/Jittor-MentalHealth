import jittor as jt
# ==========================================
# 2. Jittor 实现
# ==========================================
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        # Jittor 不需要显式指定 device，它会自动管理（通过 jt.flags.use_cuda 控制）
        mask_shape = [B, 1, L, L]
        # jt.triu 的参数是 diagonal，与 torch 一致
        # Jittor 中通常使用 float 或 int 也可以做 mask，这里显式转 bool
        self._mask = jt.triu(jt.ones(mask_shape), diagonal=1).bool()

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 确保输入是 Jittor 变量
        if not isinstance(index, jt.Var):
            index = jt.array(index)
            
        S = scores.shape[-1] # scores 的最后一维
        
        # 1. 创建基础 Mask [L, S]
        _mask = jt.ones((L, S)).triu(1).bool()
        
        # 2. 扩展维度 [B, H, L, S]
        # Jittor 的 expand 行为与 PyTorch 一致，但通常先需要 unsqueeze
        _mask_ex = _mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
        
        # 3. 构造索引
        # Jittor 支持类似 PyTorch 的高级索引，但为了稳健性，显式构造索引形状
        # torch.arange(B)[:, None, None] -> [B, 1, 1]
        b_idx = jt.arange(B).unsqueeze(1).unsqueeze(2)
        # torch.arange(H)[None, :, None] -> [1, H, 1]
        h_idx = jt.arange(H).unsqueeze(0).unsqueeze(2)
        
        # 4. 执行索引
        # index 是 [B, H, Sample_L]，作为第2维(L维度)的索引
        # Jittor 会自动广播 b_idx 和 h_idx 以匹配 index 的形状
        indicator = _mask_ex[b_idx, h_idx, index, :]
        
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask