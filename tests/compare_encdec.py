import torch
import torch.nn as nn
import numpy as np
import jittor as jt
import warnings

# ================= MOCK CLASSES (为了跑通测试构建的桩) =================
class MockAttentionTorch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, attn_mask=None, tau=None, delta=None):
        # 简单返回 projection(q) 以模拟 attention 输出，保持形状不变
        return self.proj(q), None

class MockAttentionJittor(jt.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = jt.nn.Linear(d_model, d_model)
    def execute(self, q, k, v, attn_mask=None, tau=None, delta=None):
        return self.proj(q), None

# ================= UTILS =================
def copy_weights(torch_model, jt_model):
    """
    将 PyTorch 模型的权重加载到 Jittor 模型中。
    """
    torch_params = dict(torch_model.named_parameters())
    jt_params = dict(jt_model.named_parameters())

    for name, param in torch_params.items():
        if name in jt_params:
            p_np = param.detach().numpy()
            jt_params[name].update(p_np)
        else:
            # 有些比如 buffer 或者未使用参数可能不匹配，打印警告
            pass 
            # print(f"Warning: {name} not found in Jittor model")

# ================= IMPORTS =================
# 假设文件结构如下：
# layers/Transformer_EncDec.py (Torch版本)
# layers_jt/Transformer_EncDec.py (Jittor版本)

from layers.Transformer_EncDec import ConvLayer as TorchConvLayer 
from layers_jt.Transformer_EncDec import ConvLayer as JtConvLayer

from layers.Transformer_EncDec import EncoderLayer as TorchEncoderLayer
from layers_jt.Transformer_EncDec import EncoderLayer as JtEncoderLayer

from layers.Transformer_EncDec import Encoder as TorchEncoder
from layers_jt.Transformer_EncDec import Encoder as JtEncoder

from layers.Transformer_EncDec import DecoderLayer as TorchDecoderLayer
from layers_jt.Transformer_EncDec import DecoderLayer as JtDecoderLayer

# 新增 Decoder 的引用
from layers.Transformer_EncDec import Decoder as TorchDecoder
from layers_jt.Transformer_EncDec import Decoder as JtDecoder


# ================= TEST SCRIPT =================
def run_test():
    # Settings
    B, L, D = 2, 32, 16
    d_ff = 32
    
    # Common Inputs
    np_input = np.random.randn(B, L, D).astype(np.float32)
    t_in = torch.from_numpy(np_input)
    j_in = jt.array(np_input)

    # --- 1. Test ConvLayer ---
    print("\n--- Testing ConvLayer ---")
    t_conv = TorchConvLayer(D)
    j_conv = JtConvLayer(D)
    copy_weights(t_conv, j_conv)
    
    t_conv.eval(); j_conv.eval()
    
    t_out = t_conv(t_in).detach().numpy()
    j_out = j_conv(j_in).numpy()
    
    diff = np.max(np.abs(t_out - j_out))
    print(f"ConvLayer Max Diff: {diff:.6e} {'(PASS)' if diff < 1e-5 else '(FAIL)'}")

    # --- 2. Test EncoderLayer ---
    print("\n--- Testing EncoderLayer ---")
    t_enc_layer = TorchEncoderLayer(MockAttentionTorch(D), D, d_ff=d_ff)
    j_enc_layer = JtEncoderLayer(MockAttentionJittor(D), D, d_ff=d_ff)
    copy_weights(t_enc_layer, j_enc_layer)
    
    t_enc_layer.eval(); j_enc_layer.eval()
    
    t_out, _ = t_enc_layer(t_in)
    j_out, _ = j_enc_layer(j_in)
    
    diff = np.max(np.abs(t_out.detach().numpy() - j_out.numpy()))
    print(f"EncoderLayer Max Diff: {diff:.6e} {'(PASS)' if diff < 1e-5 else '(FAIL)'}")

    # --- 3. Test Full Encoder ---
    print("\n--- Testing Full Encoder ---")
    # 2 Layers + 1 Conv
    t_layers = [TorchEncoderLayer(MockAttentionTorch(D), D, d_ff=d_ff) for _ in range(2)]
    t_convs = [TorchConvLayer(D)] 
    t_encoder = TorchEncoder(t_layers, t_convs, norm_layer=nn.LayerNorm(D))
    
    j_layers = [JtEncoderLayer(MockAttentionJittor(D), D, d_ff=d_ff) for _ in range(2)]
    j_convs = [JtConvLayer(D)]
    j_encoder = JtEncoder(j_layers, j_convs, norm_layer=jt.nn.LayerNorm(D))
    
    copy_weights(t_encoder, j_encoder)
    t_encoder.eval(); j_encoder.eval()
    
    t_out, _ = t_encoder(t_in)
    j_out, _ = j_encoder(j_in)
    
    diff = np.max(np.abs(t_out.detach().numpy() - j_out.numpy()))
    print(f"Encoder Max Diff: {diff:.6e} {'(PASS)' if diff < 1e-5 else '(FAIL)'}")

    # --- 4. Test DecoderLayer ---
    print("\n--- Testing DecoderLayer ---")
    t_dec_layer = TorchDecoderLayer(MockAttentionTorch(D), MockAttentionTorch(D), D, d_ff=d_ff)
    j_dec_layer = JtDecoderLayer(MockAttentionJittor(D), MockAttentionJittor(D), D, d_ff=d_ff)
    copy_weights(t_dec_layer, j_dec_layer)
    t_dec_layer.eval(); j_dec_layer.eval()
    
    # Cross input (Encoder output simulation)
    np_cross = np.random.randn(B, L, D).astype(np.float32)
    t_cross = torch.from_numpy(np_cross)
    j_cross = jt.array(np_cross)
    
    t_out = t_dec_layer(t_in, t_cross)
    j_out = j_dec_layer(j_in, j_cross)
    
    diff = np.max(np.abs(t_out.detach().numpy() - j_out.numpy()))
    print(f"DecoderLayer Max Diff: {diff:.6e} {'(PASS)' if diff < 1e-5 else '(FAIL)'}")

    # --- 5. Test Full Decoder (NEW) ---
    print("\n--- Testing Full Decoder ---")
    
    # Build layers list
    # Torch Decoder: Stack of DecoderLayers + LayerNorm + Optional Projection
    t_dec_layers = [
        TorchDecoderLayer(MockAttentionTorch(D), MockAttentionTorch(D), D, d_ff=d_ff) 
        for _ in range(2)
    ]
    t_decoder = TorchDecoder(t_dec_layers, norm_layer=nn.LayerNorm(D))

    # Jittor Decoder
    j_dec_layers = [
        JtDecoderLayer(MockAttentionJittor(D), MockAttentionJittor(D), D, d_ff=d_ff) 
        for _ in range(2)
    ]
    j_decoder = JtDecoder(j_dec_layers, norm_layer=jt.nn.LayerNorm(D))

    # Copy weights
    copy_weights(t_decoder, j_decoder)
    
    # Eval mode
    t_decoder.eval()
    j_decoder.eval()

    # Forward Pass
    # Decoder inputs: x, cross, x_mask=None, cross_mask=None
    t_out = t_decoder(t_in, t_cross)
    j_out = j_decoder(j_in, j_cross)

    diff = np.max(np.abs(t_out.detach().numpy() - j_out.numpy()))
    print(f"Decoder Max Diff: {diff:.6e} {'(PASS)' if diff < 1e-5 else '(FAIL)'}")


if __name__ == "__main__":
    try:
        run_test()
    except ImportError as e:
        print("\n运行错误: 请确保 models_torch.py 和 models_jt.py 在同一目录下，且包含相应的类定义。")
        print(f"Details: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()